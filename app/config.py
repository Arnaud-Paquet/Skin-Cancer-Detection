"""Configuration for the DermaScan web app.

The production model is EfficientNetV2-S, trained on HAM10000 with a
lesion-grouped 70/15/15 split. The numbers in MODEL_INFO are evaluated on
the held-out test set, which was never used for early stopping, threshold
tuning, or temperature calibration.
"""

# ─────────────────────────────────────────────────────────────────────
# Production model
# ─────────────────────────────────────────────────────────────────────
WINNER_MODEL = 'efficientnet_v2_s'
WEIGHTS_PATH = './models/best_efficientnet_v2_s.pth'

# Number of TTA augmented views averaged at inference (plus 1 clean pass).
N_TTA_AUGMENTED = 8

# Post-hoc temperature scaling fit by LBFGS on val_df (notebook calibration
# cell). Logits are divided by this before softmax — see inference.run_tta.
# A value below 1.0 means the raw model was slightly under-confident
# (label smoothing + EMA + TTA averaging compounded to damp the softmax).
TEMPERATURE = 0.8722


# ─────────────────────────────────────────────────────────────────────
# Per-architecture input resolution (must match the training setup).
# ─────────────────────────────────────────────────────────────────────
IMG_SIZE = {
    'efficientnet_v2_s': 384,
}


# ─────────────────────────────────────────────────────────────────────
# Class metadata — order matches CLASS_MAP from the notebook.
# ─────────────────────────────────────────────────────────────────────
CODE_BY_INDEX = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# Full medical names (abbreviation is shown separately in the UI).
CLASS_NAMES = {
    'nv':    'Melanocytic nevus',
    'mel':   'Melanoma',
    'bkl':   'Benign keratosis',
    'bcc':   'Basal cell carcinoma',
    'akiec': "Actinic keratosis / Bowen's disease",
    'vasc':  'Vascular lesion',
    'df':    'Dermatofibroma',
}

# One-line clinical note for each class.
CLASS_NOTES = {
    'nv':    'Benign mole',
    'mel':   'Malignant melanocytic tumour',
    'bkl':   'Benign keratosis-like lesion',
    'bcc':   'Common malignant skin cancer',
    'akiec': 'Pre-malignant / in-situ carcinoma',
    'vasc':  'Benign vascular lesion',
    'df':    'Benign fibrous nodule',
}

# Risk stratification mirrors the notebook's risk_map. The UI maps these
# to Low risk / Moderate risk / High risk for the top-class chip — this is
# orthogonal to the melanoma-concern banner below.
RISK_MAP = {
    'nv':    'benign',
    'bkl':   'benign',
    'df':    'benign',
    'vasc':  'benign',
    'akiec': 'pre-cancerous',
    'bcc':   'malignant',
    'mel':   'malignant',
}


# ─────────────────────────────────────────────────────────────────────
# Grad-CAM target conv layer per architecture (last meaningful block).
# ─────────────────────────────────────────────────────────────────────
GRADCAM_LAYER_MAP = {
    'efficientnet_v2_s': lambda m: m.features[-1],  # last fused-MBConv block
}


# ─────────────────────────────────────────────────────────────────────
# Melanoma-concern thresholds on the calibrated p_mel.
#
#   p_mel ≥ P_MEL_HIGH       → "High concern"     (red)
#   p_mel ∈ [P_MEL_MODERATE, # → "Moderate concern" (amber)
#               P_MEL_HIGH)
#   p_mel < P_MEL_MODERATE   → "Low concern"      (green)
#
# Deliberately conservative on the low end — in a triage context, the cost
# of an unflagged melanoma far exceeds the cost of an extra precautionary
# consultation, so the moderate band starts at p_mel = 0.30.
# ─────────────────────────────────────────────────────────────────────
P_MEL_HIGH     = 0.70
P_MEL_MODERATE = 0.30


# ─────────────────────────────────────────────────────────────────────
# Strings shown in the "How it works" strip on the page.
# Numbers are on a held-out test set, never used for any model decision.
# ─────────────────────────────────────────────────────────────────────
MODEL_INFO = {
    'Architecture':   'EfficientNetV2-S · fine-tuned ConvNet',
    'Dataset':        'HAM10000 · 10,015 dermatoscopic images',
    'Classes':        '7 lesion types',
    'Top-1 accuracy': '75.2 % balanced · 0.62 mel F1 · 55.3 % mel prec (held-out test)',
}


# ─────────────────────────────────────────────────────────────────────
# Upload constraints
# ─────────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 10


# ─────────────────────────────────────────────────────────────────────
# Low-confidence indicator threshold.
# When the top-class probability stays below this value, the UI shows an
# informational banner saying multiple classes are plausible.
# With 7 classes, uniform random would give ~0.14 per class. At 0.45, the
# model is significantly above chance but not confidently committing —
# the right zone for a clinical "uncertain prediction" warning.
# ─────────────────────────────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.45
