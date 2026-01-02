import os
import traceback
from flask import Flask, render_template, request, jsonify
from src.inference.predictor import load_model, predict_image_from_bytes

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model = None


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model

    try:
        if model is None:
            model = load_model()   # lazy load (Azure-safe)

        if "file" not in request.files:
            return jsonify({"prediction": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"prediction": "Empty filename"}), 400

        if not file.mimetype.startswith("image/"):
            return jsonify({"prediction": "Invalid file type"}), 400

        image_bytes = file.read()
        prediction = predict_image_from_bytes(image_bytes, model)

        return jsonify({"prediction": prediction})

    except Exception:
        traceback.print_exc()
        return jsonify({"prediction": "Prediction failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
