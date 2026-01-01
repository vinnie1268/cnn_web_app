import os
import traceback
from flask import Flask, render_template, request, jsonify
from src.inference.predictor import load_model, predict_image_from_bytes

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# -------------------------------------------------
# Load model ONCE
# -------------------------------------------------
model = None

def init_model():
    global model
    try:
        model = load_model()
        print("Model loaded successfully")
    except Exception:
        print("Model loading failed")
        traceback.print_exc()
        model = None

init_model()

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"prediction": "Model not available"}), 500

    if "file" not in request.files:
        return jsonify({"prediction": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"prediction": "Empty filename"}), 400

    if not file.mimetype.startswith("image/"):
        return jsonify({"prediction": "Invalid file type"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"prediction": "Empty image"}), 400

    try:
        prediction = predict_image_from_bytes(image_bytes, model)
        return jsonify({"prediction": prediction})
    except Exception:
        traceback.print_exc()
        return jsonify({"prediction": "Prediction failed"}), 500


# -------------------------------------------------
# Local run
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
