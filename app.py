import streamlit as st
import pandas as pd
from model import train_model

st.title("Прогноз ризику інсульту")
st.write(
    "Додаток прогнозує ризик інсульту на основі медичних та соціальних факторів.")

model, metrics, columns, data = train_model()

st.subheader("Метрики моделі")
st.write("Accuracy:", round(metrics["accuracy"], 2))
st.write("Precision:", round(metrics["precision"], 2))
st.write("Recall:", round(metrics["recall"], 2))
st.write("F1-score:", round(metrics["f1"], 2))

st.subheader("Аналіз даних")

st.write("Кількість випадків інсульту:")
stroke_counts = data["stroke"].map({0: "Немає", 1: "Є"}).value_counts()
st.bar_chart(stroke_counts)

st.subheader("Введіть дані")

age = st.number_input("Вік", 0, 100, 50)
glucose = st.number_input("Рівень глюкози", 50.0, 300.0, 100.0)
bmi = st.number_input("Індекс маси тіла (BMI)", 10.0, 60.0, 25.0)

hypertension = st.selectbox("Гіпертонія", ["Ні", "Так"])
heart = st.selectbox("Хвороби серця", ["Ні", "Так"])

gender = st.selectbox("Стать", ["Male", "Female"])
married = st.selectbox("Одружений/а", ["Yes", "No"])
work = st.selectbox("Тип роботи", ["Private", "Self-employed", "Govt_job", "children"])
residence = st.selectbox("Місце проживання", ["Urban", "Rural"])
smoking = st.selectbox("Куріння", ["never smoked", "formerly smoked", "smokes"])

user_data = pd.DataFrame([{
    "age": age,
    "avg_glucose_level": glucose,
    "bmi": bmi,
    "hypertension": 1 if hypertension == "Так" else 0,
    "heart_disease": 1 if heart == "Так" else 0,
    "gender": gender,
    "ever_married": married,
    "work_type": work,
    "Residence_type": residence,
    "smoking_status": smoking
}])

user_data = pd.get_dummies(user_data, drop_first=True)
user_data = user_data.reindex(columns=columns, fill_value=0)

if st.button("Прогнозувати"):
    probability = model.predict_proba(user_data)[0][1]

    st.write("Ймовірність інсульту:", round(probability * 100, 2), "%")

    if probability >= 0.25:
        st.error("Підвищений ризик інсульту")
    else:
        st.success("Низький ризик інсульту")

    st.caption("Це навчальна модель і не є медичним діагнозом.")