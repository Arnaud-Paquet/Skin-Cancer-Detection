# Model weights

The trained EfficientNetV2-S checkpoint (`best_efficientnet_v2_s.pth`, ~80 MB)
is not tracked in git — it ships through GitHub Releases instead.

## Setup

Download `best_efficientnet_v2_s.pth` from the latest release:

  https://github.com/Arnaud-Paquet/Skin-Cancer-Detection/releases/latest

and drop it into this folder, then start the app:

```bash
cd ..
pip install -r requirements.txt
python app.py
```

The application reads `WEIGHTS_PATH = './models/best_efficientnet_v2_s.pth'`
from `app/config.py` and loads the checkpoint at startup.

## Model details

- Architecture: EfficientNetV2-S (torchvision), fine-tuned on HAM10000.
- Input resolution: 384 × 384.
- Calibration: temperature scaling, T = 0.8722 (fit on val_df).
- Held-out test metrics: 75.2 % balanced accuracy, 0.62 mel F1,
  2.98 % ECE.

See the main `README.md` and the training notebook for the full recipe.
