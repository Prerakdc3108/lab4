from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load Trained Model & Scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")  # Load HTML page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        data = [float(request.form[key]) for key in request.form.keys()]
        data_scaled = scaler.transform([data])  # Scale input

        # Predict Fish Weight
        prediction = model.predict(data_scaled)

        return jsonify({"Predicted Weight (g)": f"{prediction[0]:.2f}"})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
