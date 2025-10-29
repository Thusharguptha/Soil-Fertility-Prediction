from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)
model = joblib.load("soil_fertility_model.pkl")

feature_ranges = {
    'N': (6, 383),
    'P': (2.9, 125),
    'K': (11, 887),
    'ph': (0.9, 11.15),
    'ec': (0.1, 0.95),
    'oc': (0.1, 24),
    'S': (0.64, 31),
    'zn': (0.07, 42),
    'fe': (0.21, 44),
    'cu': (0.09, 3.02),
    'Mn': (0.11, 31),
    'B': (0.06, 2.82),
}

@app.route('/')
def home():
    features = list(feature_ranges.keys())
    return render_template('index.html', features=features, feature_ranges=feature_ranges)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])
        prediction = model.predict(final_input)[0]

        # Map prediction to label
        fertility_map = {
            0: "Less Fertile",
            1: "Fertile",
            2: "Highly Fertile"
        }
        fertility_label = fertility_map.get(prediction, "Unknown")

        # Return form again with prediction + feature list
        return render_template(
            'index.html',
            prediction_text=f"Soil Fertility: {fertility_label}",
            features=list(feature_ranges.keys()),
            feature_ranges=feature_ranges
        )



if __name__ == '__main__':
    app.run(debug=True)
