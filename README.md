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

Or, after activation:

```bash
source .venv/bin/activate
python app.py
```

## Controls

- `q`: quit
- `c`: clear the detected mudra sequence
- `h`: show/hide on-screen jutsu sequence guide

## Jutsu Sequences (Current Demo)

- `Fireball (Demo)`: `DOG -> TIGER -> BOAR -> RAM`
- `Water Dragon (Demo)`: `BOAR -> BIRD -> RAM -> HORSE`
- `Shadow Clone (Demo)`: `RAM -> BIRD -> DOG`

Hold each mudra for a short, stable moment before switching to the next sign.

## Detection Tips For Different Skin Tones

- Face a soft front light source (window or desk lamp) rather than overhead-only light.
- Keep your hand in the center third of the frame and avoid strong backlight.
- Use a plain background color that contrasts with your hand and sleeves.
- Stay about an arm's length from camera so fingers are clearly visible.
- If tracking drops, open/close your hand once, then restart the sequence.

## Notes

- This is a fast prototype with rule-based mudra classification, not a trained model.
- The app now applies local contrast normalization and gamma correction before landmark detection.
- Hand orientation and camera angle can affect detection accuracy.
- For production quality, capture a dataset and train a classifier over hand landmarks.
- If you have package conflicts in a global Python install, always run this app through `.venv/bin/python`.
