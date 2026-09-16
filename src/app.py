from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

# Construct an absolute path so the model can be found locally and on Render.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_pipeline.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@app.route("/")
def home():
    """Display the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form values and return a model prediction."""
    try:
        patient_data = {
            feature: float(request.form[feature])
            for feature in FEATURES
        }

        patient_df = pd.DataFrame([patient_data], columns=FEATURES)

        predicted_class = int(model.predict(patient_df)[0])
        probability = float(model.predict_proba(patient_df)[0, 1])

        if predicted_class == 1:
            prediction_text = "Higher model-estimated likelihood"
            result_class = "higher"
        else:
            prediction_text = "Lower model-estimated likelihood"
            result_class = "lower"

        return render_template(
            "result.html",
            prediction_text=prediction_text,
            probability=f"{probability * 100:.1f}%",
            result_class=result_class,
        )

    except (KeyError, TypeError, ValueError):
        return render_template(
            "index.html",
            error="Please enter a valid numeric value for every field.",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)