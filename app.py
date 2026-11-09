import os

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained CNN model
MODEL_PATH = os.path.join(app.root_path, "cnn_model.keras")
model = load_model(MODEL_PATH)

# Folder containing performance graphs
OUTPUT_FOLDER = os.path.join(app.root_path, "outputs")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/outputs/<path:filename>")
def show_graph(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "digit_image" not in request.files:
        return "Please upload an image.", 400

    file = request.files["digit_image"]

    if file.filename == "":
        return "No image selected.", 400

    try:
        # Open image and convert to grayscale
        image = Image.open(file).convert("L")

        # Convert image to NumPy array
        image_array = np.array(image, dtype=np.uint8)

        # Invert if background is light
        if image_array.mean() > 127:
            image_array = 255 - image_array

        image = Image.fromarray(image_array)

        # Find the digit's bounding box
        threshold_image = image.point(
            lambda pixel: 255 if pixel > 50 else 0
        )
        bbox = threshold_image.getbbox()

        if bbox is None:
            return "No digit detected. Please upload a clearer image.", 400

        # Crop around the digit
        image = image.crop(bbox)

        # Resize while preserving proportions
        image.thumbnail((20, 20), Image.Resampling.LANCZOS)

        # Center the digit on a black 28 x 28 canvas
        canvas = Image.new("L", (28, 28), 0)
        x = (28 - image.width) // 2
        y = (28 - image.height) // 2
        canvas.paste(image, (x, y))

        # Normalize pixel values and prepare CNN input
        image_array = np.array(canvas, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict the digit
        probabilities = model.predict(image_array, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        return render_template(
            "index.html",
            prediction=predicted_digit,
            confidence=confidence
        )

    except Exception as error:
        print("Prediction error:", error)
        return (
            "Could not process this image. Please upload a valid image file.",
            400
        )


if __name__ == "__main__":
    app.run(debug=True)