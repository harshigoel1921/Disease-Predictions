from flask import Flask, render_template, request 
import numpy as np
import joblib

app = Flask(__name__)

heart_model = joblib.load('models/heart_model.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')
liver_model = joblib.load('models/liver_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']

    if disease == 'heart':
        features = [
            request.form['age'], request.form['sex'], request.form['cp'],
            request.form['trestbps'], request.form['chol'], request.form['fbs'],
            request.form['restecg'], request.form['thalach'], request.form['exang'],
            request.form['oldpeak'], request.form['slope'], request.form['ca'],
            request.form['thal']
        ]
        model = heart_model
    elif disease == 'diabetes':
        features = [
            request.form['Pregnancies'], request.form['Glucose'],
            request.form['BloodPressure'], request.form['SkinThickness'],
            request.form['Insulin'], request.form['BMI'],
            request.form['DPF'], request.form['Age']
        ]
        model = diabetes_model
    else:
        features = [
            request.form['Age'], request.form['Gender'],
            request.form['TB'], request.form['DB'],
            request.form['AP'], request.form['ALT'],
            request.form['AST'], request.form['TP'],
            request.form['Albumin'], request.form['AGR']
        ]
        model = liver_model
    values = [float(x) for x in features]
    prediction = model.predict([values])
    result = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', prediction_text=f"Result: {result}")
if __name__ == "__main__":
    app.run(debug=True)