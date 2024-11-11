import streamlit as st
import numpy as np
import joblib
st.title('heart_failure')
st.subheader("heart_failure")
age = st.number_input("Enter the age:", min_value=0, max_value=100)

anaemia = st.radio("Choisissez une valeur :", (0, 1))
st.write(f"Valeur sélectionnée : {anaemia}")

creatinine_phosphokinase = st.number_input("Enter the creatinine_phosphokinase:", min_value=0, max_value=10000)

diabetes = st.radio("Choisissez une valeur :", (0, 1),key="diabte")
st.write(f"Valeur sélectionnée : {diabetes}")

ejection_fraction = st.number_input("Enter the ejection_fraction:", min_value=0, max_value=100)

high_blood_pressure = st.radio("Choisissez une valeur :", (0, 1),key="blood")
st.write(f"Valeur sélectionnée : {high_blood_pressure}")

platelets = st.number_input("Enter the platelets:", min_value=0, max_value=400000)

serum_creatinine = st.number_input("Enter the serum_creatinine:", min_value=0, max_value=400000)

serum_sodium = st.number_input("Enter the serum_sodium:", min_value=0, max_value=4)

sex  = st.radio("Choisissez une valeur :", (0, 1),key="sexe")
st.write(f"Valeur sélectionnée : {sex }")

smoking  = st.radio("Choisissez une valeur :", (0, 1),key="smoke")
st.write(f"Valeur sélectionnée : {smoking }")

time  = st.number_input("Enter the time :", min_value=0, max_value=10)
array=np.array([age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time ])

model_scaler = joblib.load("scaler_model.pkl")
sclaedarray=model_scaler.transform([array])
model=joblib.load("random_forest_model.pkl")
prediction = model.predict(sclaedarray)
st.info(prediction)

