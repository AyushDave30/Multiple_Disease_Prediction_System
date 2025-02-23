import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models
diabetes_model = pickle.load(
    open(
        "D:/Xpython/Web App/Multiple Disease Prediction System/diabetes_model.sav", "rb"
    )
)

heart_disease_model = pickle.load(
    open(
        "D:/Xpython/Web App/Multiple Disease Prediction System/Heart_disease_model.sav",
        "rb",
    )
)


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

# Diabetes Prediction page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    # Input fields in columns for better layout
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

    diab_diagnosis = ""

    # Prediction button
    if st.button("Diabetes Test Result"):
        try:
            # Convert inputs to float
            inputs = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]

            # Make prediction
            diabetes_prediction = diabetes_model.predict([inputs])

            if diabetes_prediction[0] == 1:
                diab_diagnosis = "The person is Diabetic."
            else:
                diab_diagnosis = "The person is Non-Diabetic."

        except ValueError:
            diab_diagnosis = "Please enter valid numeric values for all fields."

        st.success(diab_diagnosis)

# Heart Disease Prediction page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    # Input fields in 3 columns
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

    heart_diagnosis = ""

    # Prediction button
    if st.button("Heart Disease Test Result"):
        try:
            # Convert inputs to float
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

            # Make prediction
            heart_prediction = heart_disease_model.predict([inputs])

            if heart_prediction[0] == 1:
                heart_diagnosis = "The person is likely to have Heart Disease."
            else:
                heart_diagnosis = "The person is unlikely to have Heart Disease."

        except ValueError:
            heart_diagnosis = "Please enter valid numeric values for all fields."

        st.success(heart_diagnosis)
