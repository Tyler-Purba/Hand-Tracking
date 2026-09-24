#!/usr/bin/env python3
"""Live two-hand coordinates. Use Python 3.10–3.12 with MediaPipe 0.10.21.

Install in a virtual environment:
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'opencv-contrib-python==4.11.0.86' 'mediapipe==0.10.21'
    python hand_tracker_two_hands.py

Allow camera access for your terminal in macOS System Settings if prompted.
Z is wrist-relative depth scaled by image width, not physical distance.
"""

import sys
import time
from datetime import datetime

import cv2
import mediapipe as mp


TIPS = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
WINDOW = "Two-hand tracker | Q to quit"


def label(frame, text, position, color=(100, 255, 180), scale=0.5):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def main():
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
                # Handedness matches the mirrored image passed to MediaPipe.
                for index, hand in enumerate(result.multi_hand_landmarks or []):
                    side = result.multi_handedness[index].classification[0].label
                    # X/Y use image width/height; Z uses width and each hand's wrist.
                    # Preserve raw values; clamp only the on-screen label placement.
                    coordinates = [
                        (round(p.x * width), round(p.y * height), round(p.z * width))
                        for p in hand.landmark
                    ]
                    fingertips = {name: coordinates[joint] for name, joint in TIPS.items()}
                    tracked.append((side, coordinates, fingertips))

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
