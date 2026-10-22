from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import MudraCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image inference using trained mudra CNN.")
    parser.add_argument("--model", type=Path, default=Path("models/mudra_cnn.pt"), help="Trained model file.")
    parser.add_argument("--image", type=Path, required=True, help="Image path to classify.")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    payload = torch.load(args.model, map_location="cpu", weights_only=True)
    class_names = payload["class_names"]
    image_size = int(payload["image_size"])

    model = MudraCNN(num_classes=len(class_names))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    tensor = tfms(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(args.topk, len(class_names))
    values, indices = torch.topk(probs, k=topk)

    print("Predictions:")
    for value, idx in zip(values.tolist(), indices.tolist()):
        print(f"  {class_names[idx]}: {value:.4f}")


if __name__ == "__main__":
    run()
