import joblib, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

print('=== HEART MODELS ===')
heart_features = pd.DataFrame([[46,1,150,65,130,70,1,1,1,1,0]],
    columns=['Age','Gender','Height(cm)','Weight(kg)','Systolic_Blood_Pressure',
             'Diastolic_Blood_Pressure','Cholesterol','Glucose','Smoking','Alcohol','Physical_Activity'])
for m in ['logistic_regression','decision_tree','random_forest','knn','svm','gradient_boosting','adaboost']:
    try:
        model = joblib.load(f'models/heart/{m}_heart_model.pkl')
        pred = model.predict(heart_features)[0]
        prob = model.predict_proba(heart_features)[0][1]*100
        print(f'  {m}: {"Positive" if pred==1 else "Negative"} ({prob:.1f}%)')
    except Exception as e:
        print(f'  {m}: ERROR - {e}')

print()
print('=== DIABETES MODELS ===')
diab_features = pd.DataFrame([[1,46,0,0,1,28.5,5.5,90]],
    columns=['Gender','Age','Hypertension','Heart_Disease','Smoking_History','BMI','Hemoglobin_A1c_Level','Blood_Glucose_Level'])
for m in ['logistic_regression','decision_tree','random_forest','knn','svm','gradient_boosting','adaboost']:
    try:
        model = joblib.load(f'models/diabetes/{m}_model.pkl')
        pred = model.predict(diab_features)[0]
        prob = model.predict_proba(diab_features)[0][1]*100
        print(f'  {m}: {"Positive" if pred==1 else "Negative"} ({prob:.1f}%)')
    except Exception as e:
        print(f'  {m}: ERROR - {e}')

print()
print('=== KIDNEY MODELS ===')
kidney_features = pd.DataFrame([[45,80,1.015,0,0,1,1,0,0,121,36,1.2,138,4.5,15,44,7800,5.2,0,0,0,1,0,0]],
    columns=['Age','Blood_Pressure','Specific_Gravity','Albumin','Sugar','Red_Blood_Cells','Pus_Cell',
             'Pus_Cell_Clumps','Bacteria','Blood_Glucose_Random','Blood_Urea','Serum_Creatinine','Sodium',
             'Potassium','Hemoglobin','Packed_Cell_Volume','White_Blood_Cell_Count','Red_Blood_Cell_Count',
             'Hypertension','Diabetes_Mellitus','Coronary_Artery_Disease','Appetite','Pedal_Edema','Anemia'])
for m in ['logistic_regression','decision_tree','random_forest','knn','svm','gradient_boosting','adaboost']:
    try:
        model = joblib.load(f'models/kidney/{m}_kidney_model.pkl')
        pred = model.predict(kidney_features)[0]
        prob = model.predict_proba(kidney_features)[0][1]*100
        print(f'  {m}: {"Positive" if pred==1 else "Negative"} ({prob:.1f}%)')
    except Exception as e:
        print(f'  {m}: ERROR - {e}')

print()
print('=== LIVER MODELS ===')
liver_features = [[45,1,0.9,0.2,100,30,25,7.0,4.5,1.5]]
for m in ['logistic_regression','decision_tree','random_forest','knn','svm','gradient_boosting','adaboost']:
    try:
        model = joblib.load(f'models/liver/{m}_model.pkl')
        pred = model.predict(liver_features)[0]
        prob = model.predict_proba(liver_features)[0][1]*100
        print(f'  {m}: {"Positive" if pred==1 else "Negative"} ({prob:.1f}%)')
    except Exception as e:
        print(f'  {m}: ERROR - {e}')