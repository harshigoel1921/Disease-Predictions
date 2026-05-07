import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs('models', exist_ok=True)

# ---------------- HEART ----------------
heart = pd.DataFrame({
    'age':[63,37,41,56,57,62],
    'sex':[1,1,0,1,0,1],
    'cp':[3,2,1,1,0,0],
    'trestbps':[145,130,130,120,120,140],
    'chol':[233,250,204,236,354,268],
    'fbs':[1,0,0,0,0,0],
    'restecg':[0,1,0,1,1,1],
    'thalach':[150,187,172,178,163,160],
    'exang':[0,0,0,0,1,0],
    'oldpeak':[2.3,3.5,1.4,0.8,0.6,3.6],
    'slope':[0,0,2,2,2,0],
    'ca':[0,0,0,0,0,2],
    'thal':[1,2,2,2,2,2],
    'target':[1,1,1,1,1,0]
})

X = heart.drop('target', axis=1)
y = heart['target']

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'models/heart_model.pkl')

# ---------------- DIABETES ----------------
diabetes = pd.DataFrame({
    'Pregnancies':[6,1,8,1,0,5],
    'Glucose':[148,85,183,89,137,116],
    'BloodPressure':[72,66,64,66,40,74],
    'SkinThickness':[35,29,0,23,35,0],
    'Insulin':[0,0,0,94,168,0],
    'BMI':[33.6,26.6,23.3,28.1,43.1,25.6],
    'DiabetesPedigreeFunction':[0.627,0.351,0.672,0.167,2.288,0.201],
    'Age':[50,31,32,21,33,30],
    'Outcome':[1,0,1,0,1,0]
})

X = diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'models/diabetes_model.pkl')

# ---------------- LIVER ----------------
liver = pd.DataFrame({
    'Age':[65,62,62,58,72,46],
    'Gender':[0,1,1,1,1,1],
    'Total_Bilirubin':[0.7,10.9,7.3,1.0,3.9,1.8],
    'Direct_Bilirubin':[0.1,5.5,4.1,0.4,2.0,0.7],
    'Alkaline_Phosphotase':[187,699,490,182,195,208],
    'Alamine_Aminotransferase':[16,64,60,14,27,19],
    'Aspartate_Aminotransferase':[18,100,68,20,59,14],
    'Total_Protiens':[6.8,7.5,7.0,6.8,7.3,7.6],
    'Albumin':[3.3,3.2,3.3,3.4,2.4,4.4],
    'Albumin_and_Globulin_Ratio':[0.9,0.74,0.89,1.0,0.4,1.3],
    'Dataset':[1,1,1,2,1,2]
})

X = liver.drop('Dataset', axis=1)
y = liver['Dataset']

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'models/liver_model.pkl')

print("✅ Models trained successfully!")