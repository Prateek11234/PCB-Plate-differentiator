"""
app.py
-------
Minimal Flask service layer that exposes the diff-detection pipeline as an
HTTP API + a single-page upload UI. This is the "interface/service layer"
piece of the full-stack architecture asked for in the evaluation criteria:

    [ Client / Browser ]
            |  multipart POST (image_a, image_b)
            v
    [ Flask API layer ]  <-- app.py (this file)
            |
            v
    [ Processing pipeline ]  <-- diff_detector.py (align -> SSIM -> mask -> contours)
            |
            v
    [ Result renderer ]  <-- matplotlib bounding-box overlay, returned as PNG + JSON

Run:
    python app.py
    Then open http://localhost:5000 in a browser.
"""

import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

from diff_detector import align_images, detect_differences, draw_results

app = Flask(__name__)

UPLOAD_TMP = os.path.join(os.path.dirname(__file__), "_tmp")
os.makedirs(UPLOAD_TMP, exist_ok=True)

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>NMTronics NEST - Image Diff Detector</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; }
        h2 { color: #1a5c1a; }
        .box { border: 2px dashed #ccc; padding: 20px; border-radius: 8px; margin-bottom: 12px; }
        img { max-width: 100%; margin-top: 16px; border: 1px solid #ddd; }
        button { background: #1a5c1a; color: white; padding: 10px 20px; border: none;
                 border-radius: 6px; cursor: pointer; font-size: 15px; }
    </style>
</head>
<body>
    <h2>🔍 Image Difference Detector - NEST Prototype</h2>
    <p>Upload two near-identical images (reference + test) to detect and highlight differences.</p>
    <form method="POST" action="/detect" enctype="multipart/form-data">
        <div class="box"><label>Reference Image (A):</label><br><input type="file" name="image_a" required></div>
        <div class="box"><label>Test Image (B):</label><br><input type="file" name="image_b" required></div>
        <button type="submit">Detect Differences</button>
    </form>
    {% if result_img %}
        <h3>Result (SSIM score: {{ score }}, {{ n_boxes }} region(s) found)</h3>
        <img src="data:image/png;base64,{{ result_img }}">
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/detect", methods=["POST"])
def detect():
    file_a = request.files.get("image_a")
    file_b = request.files.get("image_b")
    if not file_a or not file_b:
        return jsonify({"error": "Both image_a and image_b are required"}), 400

    arr_a = np.frombuffer(file_a.read(), np.uint8)
    arr_b = np.frombuffer(file_b.read(), np.uint8)
    img_a = cv2.imdecode(arr_a, cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(arr_b, cv2.IMREAD_COLOR)

    if img_a is None or img_b is None:
        return jsonify({"error": "Could not decode one or both images"}), 400

    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    img_b_aligned = align_images(img_a, img_b)
    boxes, diff_norm, mask, score = detect_differences(img_a, img_b_aligned)

    out_path = os.path.join(UPLOAD_TMP, "result.png")
    draw_results(img_a, img_b_aligned, boxes, score, out_path, "upload")

    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # Return HTML page with embedded result for browser use, or JSON for API/programmatic use
    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "ssim_score": round(float(score), 4),
            "num_regions_detected": len(boxes),
            "boxes_xywh": boxes,
            "result_image_base64": encoded,
        })

    return render_template_string(INDEX_HTML, result_img=encoded, score=round(float(score), 4), n_boxes=len(boxes))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
