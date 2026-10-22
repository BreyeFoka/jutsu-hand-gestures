# Mudra Composition Detector (Python Prototype)

This is a real-time webcam prototype that:

- detects hand landmarks with MediaPipe
- classifies simplified mudras from finger states
- composes mudra sequences and triggers demo jutsu names

## Demo Jutsu Library

Current demo mappings in `mudra_engine.py`:

- `DOG -> TIGER -> BOAR -> RAM` -> `Fireball (Demo)`
- `BOAR -> BIRD -> RAM -> HORSE` -> `Water Dragon (Demo)`
- `RAM -> BIRD -> DOG` -> `Shadow Clone (Demo)`

You can edit these sequences in `JutsuComposer.jutsu_library`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python app.py
```

Or specify model/confidence explicitly:

```bash
.venv/bin/python app.py --model models/mudra_cnn.pt --confidence 0.55
```

Hand ROI mode (recommended):

- App first detects a hand region and runs CNN on that crop.
- If no hand is found, it falls back to a center crop.
- Tune crop padding with `--roi-padding` (default `0.20`).

Or, after activation:

```bash
source .venv/bin/activate
python app.py
```

## Controls

- `q`: quit
- `c`: clear the detected mudra sequence
- `h`: show/hide on-screen jutsu sequence guide

The realtime app now uses your trained CNN for frame classification.

## Jutsu Sequences (Current Demo)

- `Fireball (Demo)`: `DOG -> TIGER -> BOAR -> RAM`
- `Water Dragon (Demo)`: `BOAR -> BIRD -> RAM -> HORSE`
- `Shadow Clone (Demo)`: `RAM -> BIRD -> DOG`

Hold each mudra for a short, stable moment before switching to the next sign.

Note: The demo jutsu composition currently uses this subset: `BIRD`, `BOAR`, `DOG`, `HORSE`, `RAM`, `TIGER`.

## Detection Tips For Different Skin Tones

- Face a soft front light source (window or desk lamp) rather than overhead-only light.
- Keep your hand in the center third of the frame and avoid strong backlight.
- Use a plain background color that contrasts with your hand and sleeves.
- Stay about an arm's length from camera so fingers are clearly visible.
- If tracking drops, open/close your hand once, then restart the sequence.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover the hand-sign rules and jutsu sequence matching in `mudra_engine.py`, without a camera. GitHub Actions runs them on every push.

## Notes

- This is a fast prototype with rule-based mudra classification, not a trained model.
- The app now applies local contrast normalization and gamma correction before landmark detection.
- Hand orientation and camera angle can affect detection accuracy.
- For production quality, capture a dataset and train a classifier over hand landmarks.
- If you have package conflicts in a global Python install, always run this app through `.venv/bin/python`.

## CNN Training (Image Dataset)

Your dataset format is supported directly:

- `dataset/train/<class_name>/*.jpg|png`
- `dataset/test/<class_name>/*.jpg|png`

Detected classes in your current dataset:

- `bird`, `boar`, `dog`, `dragon`, `hare`, `horse`, `monkey`, `ox`, `ram`, `rat`, `snake`, `tiger`, `zero`

Install ML dependencies:

```bash
.venv/bin/pip install -r requirements-ml.txt
```

This uses CPU-only PyTorch wheels to avoid large CUDA downloads.

Train CNN:

```bash
.venv/bin/python ml/train_cnn.py --dataset dataset --epochs 30 --batch-size 32
```

Outputs:

- `models/mudra_cnn.pt` (weights + class names + image size)
- `models/mudra_cnn.json` (metadata and metrics)

Run single-image inference:

```bash
.venv/bin/python ml/infer_cnn.py --model models/mudra_cnn.pt --image dataset/test/tiger/1.jpg
```
