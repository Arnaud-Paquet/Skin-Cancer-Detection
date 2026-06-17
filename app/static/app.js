(() => {
    'use strict';

    const dropZone           = document.getElementById('drop-zone');
    const sampleStrip        = document.getElementById('sample-strip');
    const fileInput          = document.getElementById('file-input');
    const loadingEl          = document.getElementById('loading');
    const errorEl            = document.getElementById('error');
    const resultCard         = document.getElementById('result-card');
    const resultImage        = document.getElementById('result-image');
    const toggleBtn          = document.getElementById('toggle-overlay');
    const toggleLabel        = document.getElementById('toggle-label');
    const topFull            = document.getElementById('top-class-full');
    const topAbbr            = document.getElementById('top-class-abbr');
    const topNote            = document.getElementById('top-class-note');
    const confidencePct      = document.getElementById('confidence-pct');
    const riskPill           = document.getElementById('risk-pill');
    const rankedList         = document.getElementById('ranked-list');
    const resetBtn           = document.getElementById('reset-btn');
    const concernBanner      = document.getElementById('concern-banner');
    const concernLevel       = document.getElementById('concern-level');
    const concernPmel        = document.getElementById('concern-pmel');
    const concernDescription = document.getElementById('concern-description');
    const lowConfBanner      = document.getElementById('lowconf-banner');
    const lowConfPct         = document.getElementById('lowconf-pct');

    /* ── Risk bucket (top-class chip) → label + Tailwind colors ──── */
    const RISK_DISPLAY = {
        'benign':        { label: 'Low risk',      text: 'text-risk-low',  bg: 'bg-risk-low-bg',  dot: 'bg-risk-low'  },
        'pre-cancerous': { label: 'Moderate risk', text: 'text-risk-mod',  bg: 'bg-risk-mod-bg',  dot: 'bg-risk-mod'  },
        'malignant':     { label: 'High risk',     text: 'text-risk-high', bg: 'bg-risk-high-bg', dot: 'bg-risk-high' },
    };

    /* ── Melanoma-concern band (calibrated p_mel) → label + colors ── */
    const CONCERN_DISPLAY = {
        'low':      { label: 'Low concern',      text: 'text-risk-low',  bg: 'bg-risk-low-bg',  border: 'border-risk-low'  },
        'moderate': { label: 'Moderate concern', text: 'text-risk-mod',  bg: 'bg-risk-mod-bg',  border: 'border-risk-mod'  },
        'high':     { label: 'High concern',     text: 'text-risk-high', bg: 'bg-risk-high-bg', border: 'border-risk-high' },
    };

    /* ── Toggle state — held client-side, no extra round-trip ───── */
    let _originalDataURL = null;   // raw uploaded image
    let _overlayDataURL  = null;   // overlay base64 returned by /predict

    /* ── DOM helpers ─────────────────────────────────────────────── */
    // The .hidden Tailwind class sets display:none; result/loading both use
    // flex layout when visible, so toggle the flex class explicitly.
    function show(el, asFlex = false) {
        el.classList.remove('hidden');
        if (asFlex) el.classList.add('flex');
    }
    function hide(el) {
        el.classList.add('hidden');
        el.classList.remove('flex');
    }

    function reset() {
        hide(resultCard);
        hide(concernBanner);
        hide(lowConfBanner);
        hide(loadingEl);
        hide(errorEl);
        errorEl.textContent = '';
        fileInput.value = '';
        rankedList.innerHTML = '';
        document.getElementById('concern-label').textContent = 'Melanoma concern';
        _originalDataURL = null;
        _overlayDataURL  = null;
        resultImage.dataset.showing = 'overlay';
        toggleLabel.textContent = 'Show original';
        show(dropZone, true);
        show(sampleStrip, true);
    }

    function showError(msg) {
        hide(loadingEl);
        hide(resultCard);
        hide(concernBanner);
        hide(lowConfBanner);
        errorEl.textContent = msg;
        show(errorEl);
        show(dropZone, true);
        show(sampleStrip, true);
    }

    /* ── Network ─────────────────────────────────────────────────── */
    async function predict(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (JPEG or PNG).');
            return;
        }

        // Capture the original as a data URL so the toggle can swap it back in
        // without a round-trip to the server.
        _originalDataURL = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });

        hide(dropZone);
        hide(sampleStrip);
        hide(resultCard);
        hide(concernBanner);
        hide(lowConfBanner);
        hide(errorEl);
        show(loadingEl, true);

        const formData = new FormData();
        formData.append('image', file);

        try {
            const r = await fetch('/predict', { method: 'POST', body: formData });
            if (!r.ok) {
                const err = await r.json().catch(() => ({}));
                throw new Error(err.error || `Server returned ${r.status}`);
            }
            render(await r.json());
        } catch (e) {
            showError(`Inference failed: ${e.message}`);
        }
    }

    /* ── Render ──────────────────────────────────────────────────── */
    function render(data) {
        // Store both images so the toggle can swap between them. Default
        // view is the Grad-CAM overlay; the toggle button label reflects
        // what the user would see on next click.
        _overlayDataURL = data.overlay_b64;
        resultImage.src = _overlayDataURL;
        resultImage.dataset.showing = 'overlay';
        toggleLabel.textContent = 'Show original';

        topFull.textContent = data.top_class.full;
        topAbbr.textContent = `(${data.top_class.abbr})`;
        topNote.textContent = data.top_class.note;

        const pct = data.confidence * 100;
        confidencePct.textContent = pct.toFixed(1);

        // Risk pill — wipe previous Tailwind color classes, then apply the new ones.
        const risk = RISK_DISPLAY[data.risk_bucket] || RISK_DISPLAY['benign'];
        riskPill.className = `inline-flex items-center gap-2 font-semibold text-[13px] px-3 py-1 rounded-full ${risk.text} ${risk.bg}`;
        riskPill.innerHTML = `<span class="w-2 h-2 rounded-full ${risk.dot}"></span>${risk.label}`;

        // Melanoma concern banner — driven by the calibrated p_mel, not the
        // top-class verdict. Color-coded with the same palette as the risk pill.
        const concern = CONCERN_DISPLAY[data.melanoma_concern.level] || CONCERN_DISPLAY['low'];
        concernBanner.className =
            `fade-in mb-4 rounded-card border px-5 py-4 flex items-start gap-3 ` +
            `${concern.bg} ${concern.text} ${concern.border}`;
        concernLevel.textContent = concern.label;
        concernPmel.textContent = `p(melanoma) = ${(data.melanoma_concern.p_mel * 100).toFixed(1)}%`;
        concernDescription.textContent = data.melanoma_concern.description;

        // BCC override — p_mel is low when BCC is predicted, but BCC is malignant.
        // Escalate to high concern and relabel the banner so it doesn't mislead.
        if (data.top_class.abbr === 'bcc') {
            const bccConcern = CONCERN_DISPLAY['high'];
            concernBanner.className =
                `fade-in mb-4 rounded-card border px-5 py-4 flex items-start gap-3 ` +
                `${bccConcern.bg} ${bccConcern.text} ${bccConcern.border}`;
            document.getElementById('concern-label').textContent = 'Cancer concern';
            concernLevel.textContent   = 'High concern';
            concernPmel.textContent    = `p(BCC) = ${(data.confidence * 100).toFixed(1)}%`;
            concernDescription.textContent =
                'Basal cell carcinoma is a malignant skin cancer. Dermatology consultation recommended.';
        }

        // Low-confidence banner — orthogonal to the concern band. Shown when the
        // top-class probability stays below the threshold set in config.py.
        if (data.confidence < window.LOW_CONFIDENCE_THRESHOLD) {
            lowConfPct.textContent = pct.toFixed(1) + '%';
            show(lowConfBanner);
        } else {
            hide(lowConfBanner);
        }

        // Ranked list — top row gets emphasis (accent bar + bold name).
        rankedList.innerHTML = '';
        data.ranked.forEach((row, i) => {
            const isTop = i === 0;
            const pPct  = (row.p * 100).toFixed(1);

            const item = document.createElement('div');
            item.className = `ds-tooltip grid items-center gap-2.5 ${isTop ? '' : 'opacity-90'}`;
            item.style.gridTemplateColumns = '140px 1fr 48px';
            item.innerHTML = `
                <span class="ds-tip">${row.note}</span>
                <span class="text-[12.5px] truncate ${isTop ? 'text-ink font-semibold' : 'text-ink-soft'}">
                    ${row.full}
                    <span class="font-mono text-[10.5px] text-ink-mute">${row.code}</span>
                </span>
                <span class="relative h-1.5 rounded bg-line-soft overflow-hidden">
                    <span class="absolute inset-y-0 left-0 rounded transition-[width] duration-500 ease-out
                                 ${isTop ? 'bg-accent' : 'bg-ink-mute'}"
                          style="width: ${pPct}%"></span>
                </span>
                <span class="font-mono text-[12px] text-ink-soft text-right">${pPct}</span>
            `;
            rankedList.appendChild(item);
        });

        hide(loadingEl);
        show(concernBanner);
        show(resultCard);
    }

    /* ── Grad-CAM toggle ─────────────────────────────────────────── */
    toggleBtn.addEventListener('click', () => {
        if (!_originalDataURL || !_overlayDataURL) return;
        if (resultImage.dataset.showing === 'overlay') {
            resultImage.src = _originalDataURL;
            resultImage.dataset.showing = 'original';
            toggleLabel.textContent = 'Show heatmap';
        } else {
            resultImage.src = _overlayDataURL;
            resultImage.dataset.showing = 'overlay';
            toggleLabel.textContent = 'Show original';
        }
    });

    /* ── Sample image strip — click any thumbnail to feed predict() ─ */
    document.querySelectorAll('.sample-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const cls = btn.dataset.sample;
            const url = `/static/samples/sample_${cls}.jpg`;
            try {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const blob = await response.blob();
                const file = new File([blob], `sample_${cls}.jpg`, { type: 'image/jpeg' });
                predict(file);
            } catch (e) {
                showError(`Could not load sample: ${e.message}`);
            }
        });
    });

    /* ── Click to browse ─────────────────────────────────────────── */
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) predict(e.target.files[0]);
    });

    /* ── Drag and drop ───────────────────────────────────────────── */
    ['dragenter', 'dragover'].forEach((ev) => {
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.add('border-accent', 'bg-accent-soft');
        });
    });
    ['dragleave', 'drop'].forEach((ev) => {
        dropZone.addEventListener(ev, (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-accent', 'bg-accent-soft');
        });
    });
    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length) predict(e.dataTransfer.files[0]);
    });

    resetBtn.addEventListener('click', reset);
})();
