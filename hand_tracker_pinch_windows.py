#!/usr/bin/env python3
"""Two-hand tracking with horizontal swipes to switch macOS Spaces. Use Python 3.10–3.12 with MediaPipe 0.10.21.

Install in a virtual environment:
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'opencv-contrib-python==4.11.0.86' 'mediapipe==0.10.21' pyautogui
    python hand_tracker_pinch_windows.py

In System Settings > Privacy & Security, enable Camera and Accessibility
for the terminal/app running Python, then restart it. Ensure Control+Left/Right
work manually (Keyboard > Keyboard Shortcuts > Mission Control).
Swipe left in the mirrored view -> Control+Right; swipe right -> Control+Left.
Either hand can trigger. Return your hand during the one-second cooldown.
Q quits when the preview is focused; Ctrl+C in the terminal also quits.
Move the mouse to a screen corner to activate PyAutoGUI's fail-safe.
Run with --dry-run to detect gestures without sending shortcuts.
Gestures pause when hand boxes overlap, handedness is uncertain, or tracking
jumps. Separate hands and allow 0.35 seconds of consistent tracking to rearm.
Raw landmark drawings/data are still shown and may jitter during occlusion.
Click the window you want to move, show an open hand, then pinch thumb and
index together for 0.12 seconds. Move your pinched hand; open it to release.
The target is the focused window at grab time, not the window under your hand.
Window dragging uses native Accessibility (no extra dependency), is limited
to the primary display, and requires a normal movable, non-full-screen window.
Swipes are suppressed while pinching. Tracking ambiguity releases the grab.
Z is wrist-relative depth scaled by image width, not physical distance.
"""

import sys
import math
import time
from datetime import datetime
from collections import deque

import cv2
import mediapipe as mp


TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
WINDOW = "Pinch to move windows | Swipe for Spaces | Q to quit"
SWIPE_DISTANCE = 0.25  # Fraction of frame width.
MAX_VERTICAL_DRIFT = 0.10  # Fraction of frame height, across the entire path.
SWIPE_SECONDS = 0.5
COOLDOWN_SECONDS = 1.0


class MacWindowMover:
    """Move a retained focused AX window without clicking or dragging its contents."""

    def __init__(self):
        import ctypes as C
        self.C = C
        class Point(C.Structure):
            _fields_ = [("x", C.c_double), ("y", C.c_double)]
        self.Point = Point
        self.ax = C.CDLL('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices')
        self.cf = C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        def bind(lib, name, result, args):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = result, args
        P, I, B = C.c_void_p, C.c_int, C.c_bool
        bind(self.cf, 'CFStringCreateWithCString', P, [P, C.c_char_p, C.c_uint32])
        bind(self.cf, 'CFRelease', None, [P])
        bind(self.cf, 'CFBooleanGetValue', B, [P])
        bind(self.ax, 'AXIsProcessTrusted', B, [])
        bind(self.ax, 'AXUIElementCreateSystemWide', P, [])
        bind(self.ax, 'AXUIElementCopyAttributeValue', I, [P, P, C.POINTER(P)])
        bind(self.ax, 'AXUIElementIsAttributeSettable', I, [P, P, C.POINTER(B)])
        bind(self.ax, 'AXUIElementSetAttributeValue', I, [P, P, P])
        bind(self.ax, 'AXUIElementSetMessagingTimeout', I, [P, C.c_float])
        bind(self.ax, 'AXValueGetValue', B, [P, I, P])
        bind(self.ax, 'AXValueCreate', P, [I, P])
        self.window = None
        self.names = {}

    def name(self, text):
        if text not in self.names:
            self.names[text] = self.cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)
        return self.names[text]

    def get(self, element, attribute):
        value = self.C.c_void_p()
        error = self.ax.AXUIElementCopyAttributeValue(element, self.name(attribute), self.C.byref(value))
        if error or not value.value:
            raise RuntimeError(f'Cannot read {attribute} (Accessibility error {error})')
        return value.value

    def grab(self):
        self.release()
        if not self.ax.AXIsProcessTrusted():
            raise RuntimeError('Enable Accessibility for your terminal/app, then restart it')
        system = self.ax.AXUIElementCreateSystemWide()
        app = None
        try:
            self.ax.AXUIElementSetMessagingTimeout(system, 0.15)
            app = self.get(system, 'AXFocusedApplication')
            self.window = self.get(app, 'AXFocusedWindow')
            self.ax.AXUIElementSetMessagingTimeout(self.window, 0.15)
            try:
                full = self.get(self.window, 'AXFullScreen')
            except RuntimeError:
                full = None  # Not every app exposes this optional attribute.
            if full:
                try:
                    if self.cf.CFBooleanGetValue(full):
                        raise RuntimeError('Exit full-screen mode before grabbing this window')
                finally:
                    self.cf.CFRelease(full)
            writable = self.C.c_bool()
            error = self.ax.AXUIElementIsAttributeSettable(self.window, self.name('AXPosition'), self.C.byref(writable))
            if error or not writable.value:
                raise RuntimeError('This focused window does not support moving')
            value = self.get(self.window, 'AXPosition')
            try:
                point = self.Point()
                if not self.ax.AXValueGetValue(value, 1, self.C.byref(point)):
                    raise RuntimeError('Cannot read window position')
                return point.x, point.y
            finally:
                self.cf.CFRelease(value)
        except Exception:
            self.release()
            raise
        finally:
            if app:
                self.cf.CFRelease(app)
            self.cf.CFRelease(system)

    def move(self, x, y):
        point = self.Point(x, y)
        value = self.ax.AXValueCreate(1, self.C.byref(point))
        if not value:
            raise RuntimeError('Cannot create window position')
        try:
            error = self.ax.AXUIElementSetAttributeValue(self.window, self.name('AXPosition'), value)
            if error:
                raise RuntimeError(f'Window move failed (Accessibility error {error})')
        finally:
            self.cf.CFRelease(value)

    def release(self):
        if self.window:
            self.cf.CFRelease(self.window)
            self.window = None

    def close(self):
        self.release()
        for value in self.names.values():
            self.cf.CFRelease(value)
        self.names.clear()


class PinchDrag:
    """Open-to-pinch transition, hysteresis, and exclusive gesture ownership."""
    def __init__(self, mover, screen_size):
        self.mover = mover
        self.screen = screen_size
        self.armed = set()
        self.owner = None
        self.pending = None
        self.since = 0.0
        self.filtered = None
        self.last_time = 0.0
        self.block_until = 0.0
        self.status = 'Open fingers, then pinch thumb + index to grab'

    def release(self, now):
        if self.mover:
            self.mover.release()
        self.owner = self.pending = None
        self.armed.clear()
        self.block_until = now + 0.4

    def update(self, now, tracked, width, height, tracking_ok):
        states = {}
        for side, joints, _ in tracked:
            distance = lambda a, b: math.hypot(joints[a][0]-joints[b][0], joints[a][1]-joints[b][1])
            scale = max(distance(0, 9), distance(5, 17), 1.0)
            ratio = distance(4, 8) / scale
            center = ((joints[4][0]+joints[8][0])/(2*width),
                      (joints[4][1]+joints[8][1])/(2*height))
            states[side] = (ratio, center)
        if not tracking_ok:
            self.release(now)
            self.status = 'Grab paused: waiting for stable tracking'
            return True
        self.armed.intersection_update(states)
        if self.owner:
            if self.owner not in states or states[self.owner][0] > 0.48:
                self.release(now)
                self.status = 'Released - open fingers before grabbing again'
                return True
            center = states[self.owner][1]
            # Time-based smoothing; relative movement avoids a jump on initial grab.
            alpha = 1 - math.exp(-max(0, now-self.last_time)/0.05)
            self.filtered = tuple(a + alpha*(b-a) for a,b in zip(self.filtered, center))
            self.last_time = now
            x = self.origin[0] + (self.filtered[0]-self.anchor[0])*self.screen[0]*1.6
            y = self.origin[1] + (self.filtered[1]-self.anchor[1])*self.screen[1]*1.6
            # Keep the title-bar origin reachable on the primary display.
            x, y = max(0, min(x, self.screen[0]-100)), max(25, min(y, self.screen[1]-60))
            try:
                if self.mover:
                    self.mover.move(round(x), round(y))
                self.status = f'Grabbing with {self.owner}: ({x:.0f}, {y:.0f})'
            except RuntimeError as exc:
                self.release(now)
                self.status = str(exc)
            return True
        for side, (ratio, _) in states.items():
            if ratio > 0.55:
                self.armed.add(side)
        candidates = [side for side, (ratio, _) in states.items() if ratio < 0.30 and side in self.armed]
        if len(candidates) == 1 and now >= self.block_until:
            side = candidates[0]
            if self.pending != side:
                self.pending, self.since = side, now
            elif now - self.since >= 0.12:
                try:
                    self.origin = self.mover.grab() if self.mover else (100, 100)
                    self.owner = side
                    self.anchor = self.filtered = states[side][1]
                    self.last_time = now
                    self.pending = None
                    self.status = f'Grabbed focused window with {side}'
                except RuntimeError as exc:
                    self.release(now)
                    self.status = str(exc)
            return True
        self.pending = None
        # Suppress swipes while any hand is closed, even if it was not armed.
        return now < self.block_until or any(ratio < 0.48 for ratio, _ in states.values())


class SwipeDetector:
    """One short, predominantly unidirectional trajectory per hand."""

    def __init__(self):
        self.points = deque()

    def reset(self):
        self.points.clear()

    def update(self, now, x, y):
        # Normalized coordinates make thresholds independent of camera resolution.
        if self.points:
            t, px, py = self.points[-1]
            if now - t > 0.15 or abs(x - px) > 0.20 or abs(y - py) > 0.20:
                self.reset()  # Tracking gap or implausible jump: start a fresh path.
        self.points.append((now, x, y))
        while self.points and now - self.points[0][0] > SWIPE_SECONDS:
            self.points.popleft()
        points = list(self.points)
        # Evaluate suffixes so a brief pause before the swat does not hide it.
        for start in range(len(points) - 2):
            path = points[start:]
            dx = x - path[0][1]
            if abs(dx) < SWIPE_DISTANCE:
                continue
            if max(p[2] for p in path) - min(p[2] for p in path) > MAX_VERTICAL_DRIFT:
                continue
            direction = 1 if dx > 0 else -1
            backward = sum(max(0.0, -direction * (b[1] - a[1]))
                           for a, b in zip(path, path[1:]))
            if backward > 0.025:  # Tolerate small landmark jitter, reject reversals.
                continue
            self.reset()
            return "right" if direction > 0 else "left"
        return None



class TrackingGuard:
    """Conservatively pause gestures when hand identity/geometry is ambiguous."""

    def __init__(self):
        self.previous = {}
        self.last_time = None
        self.stable_since = None

    def update(self, now, observations):
        # Observations contain normalized coordinates and handedness confidence.
        current = {}
        boxes = []
        reason = ""
        for side, score, points in observations:
            if side in current or score < 0.8:
                reason = "Uncertain hand identity"
            xs, ys = zip(*points)
            box = (min(xs), min(ys), max(xs), max(ys))
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < 0.0001:
                reason = "Unstable hand shape"
            center = tuple(sum(points[i][axis] for i in (0, 5, 9, 13, 17)) / 5
                           for axis in (0, 1))
            current[side] = (center, area)
            boxes.append(box)
        if len(boxes) == 2:
            a, b = boxes
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1]))
            smallest = min((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
            if overlap / max(smallest, 1e-9) > 0.15:
                reason = "Hands overlap - separate them"
        if not current:
            reason = "Show a hand"
        elif current.keys() != self.previous.keys():
            reason = reason or "Hand count or identity changed"
        elif self.last_time is not None and now - self.last_time > 0.15:
            reason = reason or "Tracking interrupted"
        else:
            for side, (center, area) in current.items():
                old_center, old_area = self.previous[side]
                if (max(abs(a-b) for a, b in zip(center, old_center)) > 0.12
                        or max(area, old_area) / max(min(area, old_area), 1e-9) > 1.8):
                    reason = reason or "Tracking jumped"
        self.previous = current
        self.last_time = now
        if reason:
            self.stable_since = None
            return False, reason
        if self.stable_since is None:
            self.stable_since = now
        if now - self.stable_since < 0.35:
            return False, "Waiting for stable tracking"
        return True, "Ready | Swipe 25% width within 0.5s"


def send_swipe(direction, keyboard):
    arrow = "right" if direction == "left" else "left"
    keyboard.hotkey("ctrl", arrow, interval=0.05)
    return arrow


def label(frame, text, position, color=(100, 255, 180), scale=0.5):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def main():
    dry_run = "--dry-run" in sys.argv
    if sys.platform != "darwin" and not dry_run:
        raise RuntimeError("Space switching requires macOS. Use --dry-run for detection only.")
    keyboard = None
    if not dry_run:
        import pyautogui
        keyboard = pyautogui
    detectors = {side: SwipeDetector() for side in ("Left", "Right")}
    guard = TrackingGuard()
    mover = MacWindowMover() if not dry_run else None
    drag = PinchDrag(mover, tuple(keyboard.size()) if keyboard else (1440, 900))
    cooldown_until = 0.0
    last_gesture = "Ready: swipe horizontally"
    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe Hands requires mediapipe==0.10.21; see installation above.")

    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open camera 0. Close other camera apps and allow camera "
                "access in System Settings > Privacy & Security > Camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        hands_api = mp.solutions.hands
        drawing = mp.solutions.drawing_utils
        last_print = -float("inf")

        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped returning frames. Check its connection and permissions.")

                frame = cv2.flip(frame, 1)  # Mirror the view for natural movement.
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                tracked = []
                observations = []
                # Handedness matches the mirrored image passed to MediaPipe.
                for index, hand in enumerate(result.multi_hand_landmarks or []):
                    classification = result.multi_handedness[index].classification[0]
                    side = classification.label
                    observations.append((side, classification.score,
                                         [(p.x, p.y) for p in hand.landmark]))
                    # X/Y use image width/height; Z uses width and each hand's wrist.
                    # Preserve raw values; clamp only the on-screen label placement.
                    coordinates = [
                        (round(p.x * width), round(p.y * height), round(p.z * width))
                        for p in hand.landmark
                    ]
                    fingertips = {name: coordinates[joint] for name, joint in TIPS.items()}
                    tracked.append((side, coordinates, fingertips))

                gesture_time = time.monotonic()
                tracking_ok, tracking_status = guard.update(gesture_time, observations)
                if keyboard is not None:
                    keyboard.failSafeCheck()
                drag_busy = drag.update(gesture_time, tracked, width, height,
                                        tracking_ok and gesture_time >= cooldown_until)
                if not tracking_ok or drag_busy or gesture_time < cooldown_until:
                    for detector in detectors.values():
                        detector.reset()
                else:
                    for side, detector in detectors.items():
                        matches = [item for item in tracked if item[0] == side]
                        if len(matches) != 1:
                            detector.reset()  # Missing/ambiguous handedness cannot share a path.
                            continue
                        joints = matches[0][1]
                        # Average wrist and four knuckles for a stable palm center.
                        center_x = sum(joints[i][0] for i in (0, 5, 9, 13, 17)) / (5 * width)
                        center_y = sum(joints[i][1] for i in (0, 5, 9, 13, 17)) / (5 * height)
                        direction = detector.update(gesture_time, center_x, center_y)
                        if direction:
                            arrow = "right" if direction == "left" else "left"
                            if keyboard is not None:
                                send_swipe(direction, keyboard)
                            last_gesture = f"{side} swipe {direction} -> Ctrl+{arrow}"
                            if dry_run:
                                last_gesture += " (dry run)"
                            print(last_gesture, flush=True)
                            cooldown_until = time.monotonic() + COOLDOWN_SECONDS
                            for other in detectors.values():
                                other.reset()
                            break  # A shared cooldown prevents simultaneous hand triggers.

                # Sort for consistent display order; these are detections, not persistent IDs.
                tracked.sort(key=lambda item: (item[0], item[1][0][0]))
                for index, (side, coordinates, fingertips) in enumerate(tracked):
                    color = (100, 255, 180) if side == "Left" else (255, 190, 100)
                    for name, (x, y, z) in fingertips.items():
                        text = f"{side} {index + 1} {name}: ({x}, {y})"
                        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        position = (max(0, min(x + 10, width - tw - 4)),
                                    max(th + 4, min(y - 12, height - baseline - 4)))
                        label(frame, text, position, color=color)
                for hand in result.multi_hand_landmarks or []:
                    drawing.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)

                # Both panels show all 21 joints, with no stale data for missing hands.
                canvas = cv2.copyMakeBorder(frame, 0, max(0, 680 - height), 0, 880,
                                            cv2.BORDER_CONSTANT, value=(24, 24, 24))
                for panel in range(2):
                    left = width + panel * 440 + 12
                    if panel < len(tracked):
                        side, coordinates, _ = tracked[panel]
                        title = f"Hand {panel + 1}: {side}"
                    else:
                        coordinates = None
                        title = f"Hand {panel + 1}: Not detected"
                    label(canvas, title, (left, 28))
                    label(canvas, "ID / JOINT                 X     Y     Z*", (left, 55), scale=0.45)
                    for joint in hands_api.HandLandmark:
                        values = coordinates[joint.value] if coordinates else None
                        row = (f"{joint.value:02d} {joint.name:<17} "
                               + (f"{values[0]:5d} {values[1]:5d} {values[2]:5d}" if values else "  --    --    --"))
                        label(canvas, row, (left, 82 + joint.value * 25), scale=0.43)
                    label(canvas, "*Z: relative to this hand's wrist", (left, 625), scale=0.45)
                    label(canvas, "Q: quit | Mirrored image coordinates", (left, 650), scale=0.43)
                label(canvas, f"Tracking {len(tracked)}/2 hands", (12, 28))

                remaining = max(0.0, cooldown_until - time.monotonic())
                label(canvas, last_gesture, (12, 55))
                status = f"Cooldown: {remaining:.1f}s" if remaining else tracking_status
                label(canvas, status, (12, 82))
                label(canvas, drag.status, (12, 109), scale=0.45)
                now = time.monotonic()
                if now - last_print >= 0.5:
                    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    if tracked:
                        lines = []
                        for index, (side, _, fingertips) in enumerate(tracked):
                            lines.append(f"[{stamp}] Hand {index + 1} ({side}) " + " | ".join(
                                f"{name}: ({x}, {y}, {z})"
                                for name, (x, y, z) in fingertips.items()
                            ))
                        print("\n".join(lines), flush=True)
                    else:
                        print(f"[{stamp}] No hands detected", flush=True)
                    last_print = now

                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if mover:
            mover.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
