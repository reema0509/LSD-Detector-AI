from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Load trained model
model = None

from pathlib import Path

MODEL_PATH = Path(__file__).parent / "lsd_model.keras"

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Upload folder
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/contact", methods=["POST"])
def contact():

    name = request.form["name"]
    email = request.form["email"]
    message = request.form["message"]

    print("Name:", name)
    print("Email:", email)
    print("Message:", message)

    return redirect("/")

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    fever_days = request.form["fever_days"]
    skin_nodules = request.form["skin_nodules"]
    lymph_nodes = request.form["lymph_nodes"]
    appetite = request.form["appetite"]
    walking = request.form["walking"]
    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    model = get_model()
    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    if probability > 0.5:
        result = "🦠 Lumpy Skin Disease Detected"
        confidence = probability * 100
    else:
        result = "✅ Healthy Cow"
        confidence = (1 - probability) * 100

    return render_template(
    "result.html",
    result=result,
    confidence=round(confidence, 2),
    image=filename,
    fever_days=fever_days,
    skin_nodules=skin_nodules,
    lymph_nodes=lymph_nodes,
    appetite=appetite,
    walking=walking
)

if __name__ == "__main__":
    app.run(debug=True)