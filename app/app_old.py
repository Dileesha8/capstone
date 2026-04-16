import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, render_template_string
from src.predict_densenet_chest_xray import predict as lung_predict
from src.predict_brain import predict as brain_predict
from src.predict_khee_oa import predict as knee_predict, CLASS_TO_KL
from src.gradcam_a import make_gradcam_heatmap, generate_sidebyside_b64, get_last_conv_layer
import tensorflow as tf

try:
    from src.predict import predict_image
    print("[STARTUP] predict_image validator: LOADED ✅")
except Exception as e:
    print(f"[STARTUP ERROR] predict_image failed to load: {e}")
    raise

app = Flask(__name__)

def _prepare_img(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedAI Diagnosis - Medical Decision Support System</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --primary: #0EA5E9; --primary-dark: #0284C7; --secondary: #8B5CF6;
            --success: #10B981; --warning: #F59E0B; --danger: #EF4444;
            --dark: #0F172A; --light: #F8FAFC; --gray: #64748B; --border: #E2E8F0;
        }
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 2rem; color: var(--dark);
        }
        .container {
            background: white; border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
            max-width: 1400px; width: 100%; overflow: hidden;
            animation: slideUp 0.6s ease-out; margin: 0 auto;
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 2.5rem 2rem; text-align: center;
            position: relative; overflow: hidden;
        }
        .header::before {
            content: ''; position: absolute; top: -50%; right: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 15s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); }
        }
        .header h1 {
            color: white; font-size: 2.5rem; font-weight: 700;
            margin-bottom: 0.5rem; position: relative; z-index: 1;
        }
        .header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; position: relative; z-index: 1; }
        .module-badges {
            display: flex; justify-content: center; gap: 0.75rem;
            margin-top: 1rem; flex-wrap: wrap; position: relative; z-index: 1;
        }
        .badge {
            background: rgba(255,255,255,0.2); color: white;
            padding: 0.3rem 0.9rem; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600;
            backdrop-filter: blur(4px); border: 1px solid rgba(255,255,255,0.3);
        }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; min-height: 600px; }
        .left-panel { padding: 3rem 2.5rem; border-right: 2px solid var(--border); }
        .right-panel {
            padding: 3rem 2.5rem;
            background: linear-gradient(135deg, #F0F9FF 0%, #EDE9FE 100%);
            display: flex; align-items: flex-start; justify-content: center;
            overflow-y: auto;
        }
        .panel-title {
            font-size: 1.5rem; font-weight: 700; color: var(--dark);
            margin-bottom: 2rem; display: flex; align-items: center; gap: 0.75rem;
        }
        .form-group { margin-bottom: 1.5rem; }
        label {
            display: block; font-weight: 600; font-size: 0.95rem;
            margin-bottom: 0.75rem; color: var(--dark);
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        select {
            width: 100%; padding: 1rem 1.25rem;
            border: 2px solid var(--border); border-radius: 12px;
            font-size: 1rem; font-family: 'Space Grotesk', sans-serif;
            background: white; cursor: pointer; transition: all 0.3s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%230EA5E9' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat; background-position: right 1rem center;
            background-size: 20px; padding-right: 3rem;
        }
        select:hover { border-color: var(--primary); }
        select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(14,165,233,0.1); }
        .disease-hint { font-size: 0.82rem; color: var(--gray); margin-top: 0.5rem; min-height: 1.2em; }
        .file-upload { position: relative; width: 100%; }
        .file-upload input[type="file"] { position: absolute; opacity: 0; width: 100%; height: 100%; cursor: pointer; }
        .file-upload-label {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            padding: 2.5rem 2rem; border: 3px dashed var(--border); border-radius: 16px;
            background: var(--light); cursor: pointer; transition: all 0.3s ease;
        }
        .file-upload:hover .file-upload-label { border-color: var(--primary); background: rgba(14,165,233,0.05); }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        .file-upload-text { font-size: 1rem; font-weight: 600; color: var(--dark); margin-bottom: 0.5rem; }
        .file-upload-hint { font-size: 0.85rem; color: var(--gray); }
        .submit-btn {
            width: 100%; padding: 1.25rem 2rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white; border: none; border-radius: 12px;
            font-size: 1.1rem; font-weight: 600;
            font-family: 'Space Grotesk', sans-serif;
            cursor: pointer; transition: all 0.3s ease;
            text-transform: uppercase; letter-spacing: 1px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .submit-btn:hover  { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2); }
        .submit-btn:active { transform: translateY(0); }
        .loading-overlay {
            display: none; position: fixed; inset: 0;
            background: rgba(15,23,42,0.6); z-index: 100;
            align-items: center; justify-content: center; flex-direction: column; gap: 1rem;
        }
        .loading-overlay.active { display: flex; }
        .spinner {
            width: 56px; height: 56px; border: 5px solid rgba(255,255,255,0.2);
            border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { color: white; font-size: 1.1rem; font-weight: 600; }
        .empty-state { text-align: center; color: var(--gray); width: 100%; }
        .empty-state-icon { font-size: 5rem; margin-bottom: 1.5rem; opacity: 0.3; }
        .empty-state-text { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }
        .empty-state-hint { font-size: 1rem; opacity: 0.7; }
        .result-section { width: 100%; animation: fadeIn 0.5s ease-out; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .result-title {
            font-size: 1.75rem; font-weight: 700; color: var(--dark);
            margin-bottom: 2rem; display: flex; align-items: center; gap: 0.75rem;
        }
        .result-title::before {
            content: '✓'; display: flex; align-items: center; justify-content: center;
            width: 45px; height: 45px; background: var(--success); color: white;
            border-radius: 50%; font-size: 1.75rem; font-weight: bold;
        }
        .result-grid { display: grid; gap: 1.5rem; }
        .result-card {
            background: white; padding: 1.75rem; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 5px solid var(--primary);
            transition: transform 0.3s ease;
        }
        .result-card:hover { transform: translateX(5px); }
        .result-label {
            font-size: 0.85rem; font-weight: 600; color: var(--gray);
            text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.75rem;
        }
        .result-value {
            font-size: 1.75rem; font-weight: 700; color: var(--dark);
            font-family: 'JetBrains Mono', monospace; word-wrap: break-word;
        }
        .severity-low    { border-left-color: var(--success); }
        .severity-medium { border-left-color: var(--warning); }
        .severity-high   { border-left-color: var(--danger);  }
        .kl-badge {
            display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px;
            font-size: 0.85rem; font-weight: 700; background: #EDE9FE; color: var(--secondary);
            margin-left: 0.5rem; vertical-align: middle;
        }
        .heatmap-section {
            margin-top: 1.5rem; background: white; border-radius: 12px;
            padding: 1.25rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 5px solid var(--secondary);
        }
        .heatmap-section img { width: 100%; border-radius: 8px; margin-top: 0.75rem; }
        .heatmap-legend { font-size: 0.8rem; color: var(--gray); margin-top: 0.6rem; font-style: italic; }
        .footer {
            text-align: center; padding: 1.5rem; color: var(--gray);
            font-size: 0.85rem; border-top: 1px solid var(--border);
        }
        @media (max-width: 1024px) {
            .main-content { grid-template-columns: 1fr; }
            .left-panel { border-right: none; border-bottom: 2px solid var(--border); }
            .right-panel { min-height: 400px; }
        }
        @media (max-width: 640px) {
            .header h1 { font-size: 1.75rem; }
            .left-panel, .right-panel { padding: 2rem 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-text">Analyzing image, please wait…</div>
    </div>

    <div class="container">
        <div class="header">
            <h1>🩺 MedAI Diagnosis</h1>
            <p>AI-Powered Medical Decision Support System</p>
            <div class="module-badges">
                <span class="badge">🫁 Chest X-ray</span>
                <span class="badge">🧠 Brain MRI</span>
                <span class="badge">🦴 Knee X-ray</span>
            </div>
        </div>

        <div class="main-content">
            <div class="left-panel">
                <div class="panel-title">📤 Upload &amp; Analyze</div>
                <form method="POST" enctype="multipart/form-data" id="diagForm">
                    <div class="form-group">
                        <label for="disease">Select Disease Type</label>
                        <select name="disease" id="disease" required>
                            <option value="lung"  {% if selected == 'lung'  %}selected{% endif %}>🫁 Chest X-ray — Pneumonia Severity</option>
                            <option value="brain" {% if selected == 'brain' %}selected{% endif %}>🧠 Brain MRI — Tumor Detection</option>
                            <option value="knee"  {% if selected == 'knee'  %}selected{% endif %}>🦴 Knee X-ray — Osteoarthritis Grading</option>
                        </select>
                        <div class="disease-hint" id="diseaseHint">
                            {% if selected == 'brain' %}Upload an axial or coronal brain MRI slice.
                            {% elif selected == 'knee' %}Upload a weight-bearing AP knee X-ray for KL grading.
                            {% else %}Upload a frontal chest X-ray image (PA or AP view).{% endif %}
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Upload Medical Image</label>
                        <div class="file-upload">
                            <input type="file" name="file" accept="image/*" required id="fileInput">
                            <label for="fileInput" class="file-upload-label">
                                <div class="upload-icon">🖼️</div>
                                <div class="file-upload-text">Click to browse or drag &amp; drop</div>
                                <div class="file-upload-hint">JPG, PNG, DICOM formats</div>
                            </label>
                        </div>
                    </div>
                    <button type="submit" class="submit-btn">🔬 Analyze Image</button>
                </form>
            </div>

            <div class="right-panel">
                {% if result %}
                <div class="result-section">

                    {% if validation_error %}
                    <div style="text-align:center; padding:2rem; background:#FEF2F2; border:2px solid #EF4444; border-radius:16px;">
                        <div style="font-size:3rem; margin-bottom:1rem;">⚠️</div>
                        <div style="font-size:1.2rem; font-weight:700; color:#EF4444; margin-bottom:0.75rem;">
                            Wrong Image Type!
                        </div>
                        <div style="color:#64748B; line-height:1.6;">
                            You selected <strong>{{ expected_part | upper }}</strong>
                            but uploaded a <strong>{{ detected_part | upper }}</strong> image.<br><br>
                            Please upload the correct <strong>{{ expected_part | upper }}</strong> scan.
                        </div>
                        <div style="margin-top:1rem; font-size:0.85rem; color:#94A3B8;">
                            Classifier confidence: {{ conf_score }}%
                        </div>
                    </div>

                    {% else %}
                    <div class="result-title">Diagnosis Results</div>
                    <div class="result-grid">
                        <div class="result-card">
                            <div class="result-label">
                                {% if disease_type == 'knee' %}KL Grade / Predicted Class{% else %}Predicted Class{% endif %}
                            </div>
                            <div class="result-value">
                                {{ label }}
                                {% if disease_type == 'knee' and kl_grade is not none %}
                                    <span class="kl-badge">KL {{ kl_grade }}</span>
                                {% endif %}
                            </div>
                        </div>

                        <div class="result-card {% if score|float < 40 %}severity-low{% elif score|float < 70 %}severity-medium{% else %}severity-high{% endif %}">
                            <div class="result-label">
                                {% if disease_type == 'knee' %}Confidence Score{% else %}Severity Score{% endif %}
                            </div>
                            <div class="result-value">{{ score }}%</div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Risk Forecast</div>
                            <div class="result-value">{{ risk }}</div>
                        </div>

                        {% if disease_type == 'knee' and recommendation %}
                        <div class="result-card severity-medium">
                            <div class="result-label">Clinical Recommendation</div>
                            <div class="result-value" style="font-size:1rem; line-height:1.5;">
                                {{ recommendation }}
                            </div>
                        </div>
                        {% endif %}
                    </div>

                    {% if heatmap_b64 %}
                    <div class="heatmap-section">
                        <div class="result-label">🔍 Explainability — Grad-CAM Heatmap</div>
                        <img src="{{ heatmap_b64 }}" alt="Grad-CAM heatmap">
                        <div class="heatmap-legend">
                            🔴 Red/yellow = most influential regions &nbsp;|&nbsp; 🔵 Blue = less relevant
                        </div>
                    </div>
                    {% endif %}

                    {% endif %}

                </div>
                {% else %}
                <div class="empty-state">
                    <div class="empty-state-icon">🔬</div>
                    <div class="empty-state-text">No Results Yet</div>
                    <div class="empty-state-hint">Upload an image to see the analysis</div>
                </div>
                {% endif %}
            </div>
        </div>

        <div class="footer">
            <p>⚕️ For research and educational purposes only • Not intended for clinical diagnosis</p>
        </div>
    </div>

    <script>
        const hints = {
            lung:  "Upload a frontal chest X-ray image (PA or AP view).",
            brain: "Upload an axial or coronal brain MRI slice.",
            knee:  "Upload a weight-bearing AP knee X-ray for KL grading."
        };
        const diseaseSelect = document.getElementById('disease');
        const hintEl        = document.getElementById('diseaseHint');
        diseaseSelect.addEventListener('change', () => {
            hintEl.textContent = hints[diseaseSelect.value] || '';
        });
        const fileInput = document.getElementById('fileInput');
        const fileLabel = document.querySelector('.file-upload-label');
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                fileLabel.querySelector('.file-upload-text').textContent = e.target.files[0].name;
                fileLabel.querySelector('.upload-icon').textContent = '✅';
            }
        });
        ['dragenter','dragover','dragleave','drop'].forEach(ev =>
            fileLabel.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
        );
        ['dragenter','dragover'].forEach(ev =>
            fileLabel.addEventListener(ev, () => {
                fileLabel.style.borderColor = 'var(--primary)';
                fileLabel.style.background  = 'rgba(14,165,233,0.1)';
            })
        );
        ['dragleave','drop'].forEach(ev =>
            fileLabel.addEventListener(ev, () => {
                fileLabel.style.borderColor = 'var(--border)';
                fileLabel.style.background  = 'var(--light)';
            })
        );
        document.getElementById('diagForm').addEventListener('submit', () => {
            document.getElementById('loadingOverlay').classList.add('active');
        });
    </script>
</body>
</html>
"""

KNEE_RECOMMENDATIONS = {
    0: "No signs of OA. Continue routine check-ups.",
    1: "Doubtful narrowing. Monitor annually; lifestyle advice recommended.",
    2: "Mild OA. Consider physiotherapy and weight management.",
    3: "Moderate OA. Refer to orthopaedics; pain management indicated.",
    4: "Severe OA. Surgical consultation (e.g. TKR) strongly advised.",
}

EXPECTED_PART = {
    "lung":  "chest",
    "brain": "brain",
    "knee":  "knee",
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        disease = request.form["disease"]
        file    = request.files["file"]

        upload_path = "upload.jpg"
        file.save(upload_path)

        expected_part = EXPECTED_PART[disease]

        # ── Step 1: validate body part ──
        detected_part, conf = predict_image(upload_path)
        print(f"[VALIDATOR] Expected: {expected_part} | Detected: {detected_part} | Conf: {conf*100:.1f}%")

        if detected_part != expected_part or conf < 0.65:
            return render_template_string(
                HTML,
                result=True,
                validation_error=True,
                expected_part=expected_part,
                detected_part=detected_part,
                conf_score=f"{conf * 100:.1f}",
                label=None, score=None, risk=None,
                disease_type=disease, selected=disease,
                kl_grade=None, recommendation=None,
                heatmap_b64=None,
            )

        # ── Step 2: run disease model ──
        recommendation = None
        kl_grade       = None
        heatmap_b64    = None
        conv_layer     = None

        if disease == "lung":
            label, score, risk = lung_predict(upload_path)
            from src.predict_densenet_chest_xray import model as _model
            conv_layer = "conv5_block16_concat"   # DenseNet last block ✅

        elif disease == "brain":
            label, score, risk = brain_predict(upload_path)
            from src.predict_brain import model as _model
            conv_layer = "conv5_block16_concat"   # DenseNet last block ✅

        else:  # knee
            label, score, risk = knee_predict(upload_path)
            kl_grade       = CLASS_TO_KL.get(str(label).lower())
            recommendation = KNEE_RECOMMENDATIONS.get(kl_grade, "")
            import tf_keras
            _model     = tf_keras.models.load_model(
                r'C:/Users/91701/capstone/models/knee_oa_best.h5', compile=False)
            conv_layer = get_last_conv_layer(_model)  # knee model auto-detect

        # ── Step 3: Grad-CAM ──
        print(f"[GRADCAM] conv_layer detected: {conv_layer}")
        try:
            img_arr     = _prepare_img(upload_path)
            heatmap     = make_gradcam_heatmap(img_arr, _model, conv_layer)
            heatmap_b64 = generate_sidebyside_b64(upload_path, heatmap)
            print(f"[GRADCAM] heatmap_b64 generated OK, length: {len(heatmap_b64)}")
        except Exception as e:
            print(f"[GRADCAM ERROR] {e}")
            heatmap_b64 = None

        return render_template_string(
            HTML,
            result=True,
            validation_error=False,
            expected_part=expected_part,
            detected_part=detected_part,
            conf_score=f"{conf * 100:.1f}",
            label=label,
            score=score,
            risk=risk,
            disease_type=disease,
            selected=disease,
            kl_grade=kl_grade,
            recommendation=recommendation,
            heatmap_b64=heatmap_b64,
        )

    return render_template_string(
        HTML,
        result=False,
        validation_error=False,
        disease_type=None,
        selected='lung',
        expected_part=None,
        detected_part=None,
        conf_score=None,
        kl_grade=None,
        recommendation=None,
        heatmap_b64=None,
    )


if __name__ == "__main__":
    app.run(debug=True)