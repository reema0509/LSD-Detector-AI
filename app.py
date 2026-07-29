from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
# Load trained model


import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lsd_model.keras"

model = None

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
    from flask import send_from_directory

@app.route("/sitemap.xml")
def sitemap():
    return send_from_directory("static", "sitemap.xml")

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

    # AI Model Prediction
    loaded_model = get_model()

    prediction = loaded_model.predict(img_array)[0][0]

    # Healthy = 0
    # Infected = 1
    if prediction >= 0.5:
        result = "Infected"
        confidence = round(float(prediction) * 100, 2)
    else:
        result = "Healthy"
        confidence = round(float(1 - prediction) * 100, 2)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image=filename,
        fever_days=fever_days,
        skin_nodules=skin_nodules,
        lymph_nodes=lymph_nodes,
        appetite=appetite,
        walking=walking
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)