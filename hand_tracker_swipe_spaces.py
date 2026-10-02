#!/usr/bin/env python3
"""Two-hand tracking with horizontal swipes to switch macOS Spaces. Use Python 3.10–3.12 with MediaPipe 0.10.21.

Install in a virtual environment:
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'opencv-contrib-python==4.11.0.86' 'mediapipe==0.10.21' pyautogui
    python hand_tracker_swipe_spaces.py

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
Z is wrist-relative depth scaled by image width, not physical distance.
"""

import sys
import time
from datetime import datetime
from collections import deque

import cv2
import mediapipe as mp


TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
WINDOW = "Swipe to switch Spaces | Q to quit"
SWIPE_DISTANCE = 0.25  # Fraction of frame width.
MAX_VERTICAL_DRIFT = 0.10  # Fraction of frame height, across the entire path.
SWIPE_SECONDS = 0.5
COOLDOWN_SECONDS = 1.0


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
                if not tracking_ok or gesture_time < cooldown_until:
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
