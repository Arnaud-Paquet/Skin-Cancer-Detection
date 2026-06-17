"""Flask entry point for DermaScan.

  - /         serves the one-page UI
  - /predict  accepts a multipart image, returns JSON:
              {
                top_class:        { full, abbr, note },
                confidence:       float in [0, 1],
                risk_bucket:      'benign' | 'pre-cancerous' | 'malignant',
                melanoma_concern: { level, p_mel, description },
                ranked:           [{ code, full, p }, ...]  (all 7, sorted desc),
                overlay_b64:      'data:image/png;base64,...'
              }
"""

import base64
import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

import config
import inference


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024


# ─────────────────────────────────────────────────────────────────────
# Load the model once at startup. The Grad-CAM target layer reference
# and the calibration temperature are captured here so we don't re-resolve
# them on every request.
# ─────────────────────────────────────────────────────────────────────
print(f"[startup] Loading {config.WINNER_MODEL} from {config.WEIGHTS_PATH} ...")
MODEL, DEVICE = inference.load_model(config.WINNER_MODEL, config.WEIGHTS_PATH)
TARGET_LAYER = config.GRADCAM_LAYER_MAP[config.WINNER_MODEL](MODEL)
IMG_SIZE = config.IMG_SIZE[config.WINNER_MODEL]
TEMPERATURE = config.TEMPERATURE
MEL_INDEX = config.CODE_BY_INDEX.index('mel')

CONCERN_DESCRIPTIONS = {
    'low':      'Low melanoma concern. Routine self-monitoring.',
    'moderate': 'Moderate melanoma concern. Consider dermatology consultation.',
    'high':     'High melanoma concern. Dermatology consultation recommended.',
}

print(f"[startup] Ready on {DEVICE}. Input size: {IMG_SIZE}px. "
      f"Temperature: {TEMPERATURE:.4f}.")


def _melanoma_concern(p_mel: float) -> dict:
    """Bucket a calibrated melanoma probability into the triage band."""
    if p_mel >= config.P_MEL_HIGH:
        level = 'high'
    elif p_mel >= config.P_MEL_MODERATE:
        level = 'moderate'
    else:
        level = 'low'
    return {
        'level':       level,
        'p_mel':       p_mel,
        'description': CONCERN_DESCRIPTIONS[level],
    }


@app.route('/')
def index():
    return render_template(
        'index.html',
        model_info=config.MODEL_INFO,
        max_upload_mb=config.MAX_UPLOAD_MB,
        low_confidence_threshold=config.LOW_CONFIDENCE_THRESHOLD,
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request.'}), 400

    f = request.files['image']
    if not f.filename:
        return jsonify({'error': 'Empty file.'}), 400

    try:
        original_pil = Image.open(f.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Could not decode image: {e}'}), 400

    img_hwc = np.array(original_pil)

    # ── TTA inference with calibrated softmax ────────────────────────
    avg_probs = inference.run_tta(
        MODEL, img_hwc, IMG_SIZE, DEVICE,
        n_augmented=config.N_TTA_AUGMENTED,
        temperature=TEMPERATURE,
    )
    top_idx = int(np.argmax(avg_probs))
    top_code = config.CODE_BY_INDEX[top_idx]
    p_mel = float(avg_probs[MEL_INDEX])

    # ── Grad-CAM on the clean center-crop view (no temperature) ─────
    cam = inference.compute_gradcam(
        MODEL, img_hwc, IMG_SIZE, DEVICE,
        target_layer=TARGET_LAYER,
        target_class_idx=top_idx,
    )
    overlay_pil = inference.build_overlay(original_pil, cam, alpha=0.45)

    # ── Encode overlay as a base64 data URL ──────────────────────────
    buf = io.BytesIO()
    overlay_pil.save(buf, format='PNG')
    overlay_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    # ── Full ranked list (all 7 classes, sorted by probability desc) ─
    order = np.argsort(avg_probs)[::-1]
    ranked = [
        {
            'code': config.CODE_BY_INDEX[i],
            'full': config.CLASS_NAMES[config.CODE_BY_INDEX[i]],
            'note': config.CLASS_NOTES[config.CODE_BY_INDEX[i]],
            'p':    float(avg_probs[i]),
        }
        for i in order
    ]

    return jsonify({
        'top_class': {
            'full': config.CLASS_NAMES[top_code],
            'abbr': top_code,
            'note': config.CLASS_NOTES[top_code],
        },
        'confidence':       float(avg_probs[top_idx]),
        'risk_bucket':      config.RISK_MAP[top_code],
        'melanoma_concern': _melanoma_concern(p_mel),
        'ranked':           ranked,
        'overlay_b64':      f'data:image/png;base64,{overlay_b64}',
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': f'File exceeds {config.MAX_UPLOAD_MB} MB limit.'}), 413


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
