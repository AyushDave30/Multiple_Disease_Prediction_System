import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import time

# Load the saved models and scaler
diabetes_model = pickle.load(
    open(
        "D:/Xpython/Web App/Multiple Disease Prediction System/diabetes_model.sav", "rb"
    )
)
diabetes_scaler = pickle.load(
    open(
        "D:/Xpython/Web App/Multiple Disease Prediction System/diabetes_scaler.sav",
        "rb",
    )
)
heart_disease_model = pickle.load(
    open(
        "D:/Xpython/Web App/Multiple Disease Prediction System/Heart_disease_model.sav",
        "rb",
    )
)

# Initialize session state for results and button clicks
if "diabetes_result" not in st.session_state:
    st.session_state.diabetes_result = None

if "heart_result" not in st.session_state:
    st.session_state.heart_result = None

if "diabetes_clicked" not in st.session_state:
    st.session_state.diabetes_clicked = False

if "heart_clicked" not in st.session_state:
    st.session_state.heart_clicked = False

# App header
st.header("Welcome to the Multiple Disease Prediction System")
st.write(
    "This app uses machine learning models to predict the likelihood of **Diabetes** and **Heart Disease**. Fill in the required details and click the test result button!"
)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction"],
        icons=["activity", "heart"],
        default_index=0,
    )

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        BloodPressure = st.text_input("Blood Pressure")
        Insulin = st.text_input("Insulin Level")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")

    with col2:
        Glucose = st.text_input("Glucose Level")
        SkinThickness = st.text_input("Skin Thickness")
        BMI = st.text_input("Body Mass Index (BMI)")
        Age = st.text_input("Age")

    # Prediction button
    if st.button("Diabetes Test Result"):
        st.session_state.diabetes_clicked = True  # Track that button was clicked
        st.session_state.diabetes_result = None  # Reset result when button is clicked

    # Perform prediction only when button is clicked
    if st.session_state.diabetes_clicked:
        try:
            # Ensure all fields are filled
            if any(
                val.strip() == ""
                for val in [
                    Pregnancies,
                    Glucose,
                    BloodPressure,
                    SkinThickness,
                    Insulin,
                    BMI,
                    DiabetesPedigreeFunction,
                    Age,
                ]
            ):
                st.error("Please fill in all the fields.")
            else:
                # Convert inputs
                inputs = np.asarray(
                    [
                        float(Pregnancies),
                        float(Glucose),
                        float(BloodPressure),
                        float(SkinThickness),
                        float(Insulin),
                        float(BMI),
                        float(DiabetesPedigreeFunction),
                        float(Age),
                    ]
                ).reshape(1, -1)

                # Scale input
                inputs_scaled = diabetes_scaler.transform(inputs)

                # Predict
                diabetes_prediction = diabetes_model.predict(inputs_scaled)

                # Store result in session state
                st.session_state.diabetes_result = (
                    "The person is Diabetic."
                    if diabetes_prediction[0] == 1
                    else "The person is Non-Diabetic."
                )

                # Reset button state after prediction
                st.session_state.diabetes_clicked = False

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

    # Show the result only if available
    if st.session_state.diabetes_result:
        st.success(st.session_state.diabetes_result)


if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
        sex = st.text_input("Sex (1 = Male, 0 = Female)")
        cp = st.text_input("Chest Pain Type (cp)")
        trestbps = st.text_input("Resting Blood Pressure (trestbps)")

    with col2:
        chol = st.text_input("Serum Cholesterol (chol)")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
        restecg = st.text_input("Resting Electrocardiographic Results (restecg)")
        thalach = st.text_input("Maximum Heart Rate Achieved (thalach)")

    with col3:
        exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
        oldpeak = st.text_input("ST Depression Induced by Exercise (oldpeak)")
        slope = st.text_input("Slope of the Peak Exercise ST Segment")
        ca = st.text_input("Number of Major Vessels Colored by Fluoroscopy (ca)")
        thal = st.text_input("Thalassemia (thal)")

    # Prediction button
    if st.button("Heart Disease Test Result"):
        st.session_state.heart_clicked = True  # Track button click
        st.session_state.heart_result = None  # Reset result when button is clicked

    # Perform prediction only when button is clicked
    if st.session_state.heart_clicked:
        try:
            if any(
                val.strip() == ""
                for val in [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
            ):
                st.error("Please fill in all the fields.")
            else:
                inputs = [
                    float(age),
                    float(sex),
                    float(cp),
                    float(trestbps),
                    float(chol),
                    float(fbs),
                    float(restecg),
                    float(thalach),
                    float(exang),
                    float(oldpeak),
                    float(slope),
                    float(ca),
                    float(thal),
                ]

                # Predict
                heart_prediction = heart_disease_model.predict([inputs])

                # Store result in session state
                st.session_state.heart_result = (
                    "The person is likely to have Heart Disease."
                    if heart_prediction[0] == 1
                    else "The person is unlikely to have Heart Disease."
                )

                # Reset button state after prediction
                st.session_state.heart_clicked = False

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")

    if st.session_state.heart_result:
        st.success(st.session_state.heart_result)
