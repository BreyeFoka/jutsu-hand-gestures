from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from mudra_engine import JutsuComposer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime mudra and jutsu detector using trained CNN.")
    parser.add_argument("--model", type=Path, default=Path("models/mudra_cnn.pt"), help="Path to trained CNN model.")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum confidence to accept a mudra.")
    parser.add_argument("--window", type=int, default=8, help="Smoothing window size.")
    parser.add_argument("--stable-threshold", type=int, default=6, help="Required votes in window to commit a mudra.")
    parser.add_argument("--roi-padding", type=float, default=0.20, help="Extra padding around detected hand ROI.")
    return parser.parse_args()


def normalize_frame_for_model(frame_bgr: np.ndarray) -> np.ndarray:
    """Improve local contrast and low-light visibility for more stable frame classification."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    corrected = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 1.15
    lut_arr = np.array(
        [min(255, max(0, int(((i / 255.0) ** (1.0 / gamma)) * 255))) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(corrected, lut_arr)


class CNNMudraPredictor:
    LABEL_MAP = {
        "bird": "BIRD",
        "boar": "BOAR",
        "dog": "DOG",
        "dragon": None,
        "hare": None,
        "horse": "HORSE",
        "monkey": None,
        "ox": None,
        "ram": "RAM",
        "rat": None,
        "snake": None,
        "tiger": "TIGER",
        "zero": None,
    }

    def __init__(self, model_path: Path, confidence_threshold: float = 0.55) -> None:
        import torch
        from PIL import Image
        from torchvision import transforms

        from ml.model import MudraCNN

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train with: .venv/bin/python ml/train_cnn.py --dataset dataset"
            )

        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        class_names = payload["class_names"]
        image_size = int(payload["image_size"])

        self._torch = torch
        self._image_from_array = Image.fromarray
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MudraCNN(num_classes=len(class_names)).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

        self.transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def predict(self, frame_bgr: np.ndarray) -> Tuple[Optional[str], str, float]:
        crop = normalize_frame_for_model(frame_bgr)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = self._image_from_array(rgb)
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)

        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = self._torch.softmax(logits, dim=1)[0]

        conf, idx = self._torch.max(probs, dim=0)
        raw_label = self.class_names[int(idx.item())]
        confidence = float(conf.item())

        mapped = self.LABEL_MAP.get(raw_label.lower())
        if mapped is None or confidence < self.confidence_threshold:
            return None, raw_label, confidence
        return mapped, raw_label, confidence


class HandRegionDetector:
    def __init__(self, padding: float = 0.20) -> None:
        self.padding = max(0.0, min(0.6, padding))
        self.available = False
        self._hands = None

        try:
            import mediapipe as mp

            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                model_complexity=0,
                max_num_hands=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )
            self.available = True
        except Exception:
            self.available = False

    def _square_crop_fallback(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
        h, w = frame.shape[:2]
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return frame[y0 : y0 + side, x0 : x0 + side], (x0, y0, x0 + side, y0 + side), "center"

    def get_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
        if not self.available or self._hands is None:
            return self._square_crop_fallback(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return self._square_crop_fallback(frame)

        h, w = frame.shape[:2]
        lm = result.multi_hand_landmarks[0].landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = box_w * self.padding
        pad_y = box_h * self.padding

        x0 = int(max(0, (x_min - pad_x) * w))
        y0 = int(max(0, (y_min - pad_y) * h))
        x1 = int(min(w - 1, (x_max + pad_x) * w))
        y1 = int(min(h - 1, (y_max + pad_y) * h))

        if x1 <= x0 or y1 <= y0:
            return self._square_crop_fallback(frame)

        side = max(x1 - x0, y1 - y0)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)
        x0 = max(0, x1 - side)
        y0 = max(0, y1 - side)

        return frame[y0:y1, x0:x1], (x0, y0, x1, y1), "hand"

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()


def most_common_label(labels: Deque[str]) -> Optional[str]:
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def run() -> None:
    args = parse_args()

    predictor = CNNMudraPredictor(args.model, confidence_threshold=args.confidence)
    roi_detector = HandRegionDetector(padding=args.roi_padding)
    composer = JutsuComposer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    frame_index = 0
    active_jutsu = ""
    last_committed_mudra: Optional[str] = None
    show_help = True
    stable_window: Deque[str] = deque(maxlen=max(1, args.window))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        frame = cv2.flip(frame, 1)

        roi_frame, roi_box, roi_mode = roi_detector.get_roi(frame)
        mudra_label, raw_label, confidence = predictor.predict(roi_frame)
        if mudra_label:
            stable_window.append(mudra_label)
        else:
            stable_window.clear()

        if stable_window:
            committed_candidate = most_common_label(stable_window)
            count = stable_window.count(committed_candidate) if committed_candidate else 0

            if (
                committed_candidate
                and count >= args.stable_threshold
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
            f"Mudra: {mudra_label or '-'}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (80, 245, 120),
            2,
        )
        cv2.putText(
            frame,
            f"CNN: {raw_label} ({confidence:.2f})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 90),
            2,
        )
        cv2.putText(
            frame,
            f"ROI: {roi_mode}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (140, 240, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Sequence: {' -> '.join(composer.recent_sequence()) or '-'}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 90),
            2,
        )
        cv2.putText(
            frame,
            f"Jutsu: {active_jutsu or '-'}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (90, 180, 255),
            2,
        )
        cv2.putText(
            frame,
            "Controls: q=quit, c=clear sequence, h=toggle jutsu help",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
        )

        x0, y0, x1, y1 = roi_box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 220, 255), 2)

        if show_help:
            y0 = 210
            for i, (name, sequence) in enumerate(composer.jutsu_library.items()):
                cv2.putText(
                    frame,
                    f"{name}: {' -> '.join(sequence)}",
                    (20, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 245, 255),
                    1,
                )

        cv2.imshow("Mudra Composition Detector (CNN)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            composer.events.clear()
            active_jutsu = ""
            stable_window.clear()
        if key == ord("h"):
            show_help = not show_help

    cap.release()
    roi_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
