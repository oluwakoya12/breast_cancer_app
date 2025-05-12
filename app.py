from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load('breast_cancer_model.pkl')

# Route: Homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route: Predict
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            data = np.array(features).reshape(1, -1)
            prediction = model.predict(data)
            result = "Benign" if prediction[0] == 1 else "Malignant"
            return render_template('index.html', result=f'Prediction: {result}')
        except Exception as e:
            return render_template('index.html', result=f'Error: {str(e)}')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)