from __future__ import annotations

from collections import Counter
from math import acos, hypot
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mudra_engine import JutsuComposer, MudraClassifier


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize_frame_for_landmarks(frame_bgr):
    """Improve contrast and brightness to stabilize landmark detection in mixed lighting."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE improves local contrast without over-amplifying noise.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Mild gamma lift for low-light scenes.
    gamma = 1.15
    lut_arr = np.array(
        [min(255, max(0, int(((i / 255.0) ** (1.0 / gamma)) * 255))) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(corrected, lut_arr)


def _joint_angle(a, b, c) -> float:
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    ab_norm = max(hypot(*ab), 1e-6)
    cb_norm = max(hypot(*cb), 1e-6)
    dot = (ab[0] * cb[0] + ab[1] * cb[1]) / (ab_norm * cb_norm)
    dot = max(-1.0, min(1.0, dot))
    return acos(dot)


def finger_state_from_landmarks(hand_landmarks, handedness_label: str) -> Tuple[bool, bool, bool, bool, bool]:
    lm = hand_landmarks.landmark

    # Thumb is horizontal relative to hand orientation.
    if handedness_label == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    # Use both fingertip position and joint angle to reduce false negatives.
    def finger_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]
        angle = _joint_angle(tip, pip, mcp)
        return (tip.y < pip.y) and (angle > 2.35)

    index_up = finger_extended(8, 6, 5)
    middle_up = finger_extended(12, 10, 9)
    ring_up = finger_extended(16, 14, 13)
    pinky_up = finger_extended(20, 18, 17)

    return thumb_up, index_up, middle_up, ring_up, pinky_up


def most_common_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def run() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    classifier = MudraClassifier()
    composer = JutsuComposer()

    frame_index = 0
    stable_window: List[str] = []
    stable_window_size = 8
    stable_threshold = 6
    last_committed_mudra: Optional[str] = None
    active_jutsu = ""
    show_help = True

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            frame = cv2.flip(frame, 1)
            processed_frame = normalize_frame_for_landmarks(frame)
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            detected_mudras: List[str] = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    handed_label = handedness.classification[0].label
                    state = finger_state_from_landmarks(landmarks, handed_label)
                    mudra = classifier.classify(state)
                    if mudra:
                        detected_mudras.append(mudra)

                    mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            current_mudra = most_common_label(detected_mudras)

            if current_mudra:
                stable_window.append(current_mudra)
            else:
                stable_window.clear()

            if len(stable_window) > stable_window_size:
                stable_window = stable_window[-stable_window_size:]

            if stable_window:
                committed_candidate = most_common_label(stable_window)
                count = stable_window.count(committed_candidate) if committed_candidate else 0

                if (
                    committed_candidate
                    and count >= stable_threshold
                    and committed_candidate != last_committed_mudra
                ):
                    match = composer.add_mudra(committed_candidate, frame_index)
                    last_committed_mudra = committed_candidate
                    if match:
                        active_jutsu = f"{match.name}: {' -> '.join(match.sequence)}"
            else:
                last_committed_mudra = None

            cv2.putText(
                frame,
                f"Mudra: {current_mudra or '-'}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (80, 245, 120),
                2,
            )
            cv2.putText(
                frame,
                f"Sequence: {' -> '.join(composer.recent_sequence()) or '-'}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 90),
                2,
            )
            cv2.putText(
                frame,
                f"Jutsu: {active_jutsu or '-'}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (90, 180, 255),
                2,
            )
            cv2.putText(
                frame,
                "Controls: q=quit, c=clear sequence",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                frame,
                "h=toggle jutsu help",
                (20, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
            )

            if show_help:
                help_lines = [
                    "Fireball: DOG -> TIGER -> BOAR -> RAM",
                    "Water Dragon: BOAR -> BIRD -> RAM -> HORSE",
                    "Shadow Clone: RAM -> BIRD -> DOG",
                ]
                y0 = 140
                for i, line in enumerate(help_lines):
                    cv2.putText(
                        frame,
                        line,
                        (20, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (180, 245, 255),
                        1,
                    )

            cv2.imshow("Mudra Composition Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                composer.events.clear()
                active_jutsu = ""
            if key == ord("h"):
                show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
