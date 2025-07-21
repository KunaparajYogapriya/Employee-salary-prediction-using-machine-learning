import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
from fpdf import FPDF

# --- Dark/Light Mode Toggle ---
theme = st.sidebar.radio('Theme', ['Light', 'Dark'], index=0)
if theme == 'Dark':
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #222 !important; color: #fff !important; }
        .stButton>button { color: #222 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #fff !important; color: #222 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
st.title("Employee Salary Prediction")

# --- Always Load Data from File ---
default_path = r"Salary Data.csv"
if os.path.exists(default_path):
    df = pd.read_csv(default_path)
else:
    st.error(f"File not found at {default_path}. Please make sure the dataset is present.")
    st.stop()

df.columns = df.columns.str.strip()
cat_cols = ['Gender', 'Education Level', 'Job Title']
encoders = {}
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

scaler = StandardScaler()
df[['Age', 'Years of Experience']] = scaler.fit_transform(df[['Age', 'Years of Experience']])

df = df.dropna()
if df.shape[0] > 0:
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[['Age', 'Years of Experience', 'Salary']]))
    mask = (z_scores < 4).all(axis=1)
    if mask.sum() > 0:
        df = df[mask]
else:
    st.error("No data left after dropping missing values. Please check your dataset.")
    st.stop()

if df.shape[0] == 0:
    st.error("No data available after preprocessing. Please check your dataset or preprocessing steps.")
    st.stop()

X = df.drop('Salary', axis=1)
y = df['Salary']

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
except ValueError as e:
    st.error(f"Error during train_test_split: {e}")
    st.stop()

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Real-Time Salary Prediction Only ---
st.write("## Predict Employee Salary")

# Use session state to store prediction and PDF
if 'pdf_output' not in st.session_state:
    st.session_state['pdf_output'] = None
if 'predicted_salary' not in st.session_state:
    st.session_state['predicted_salary'] = None

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    gender = st.selectbox("Gender", encoders['Gender'].classes_)
    education = st.selectbox("Education Level", encoders['Education Level'].classes_)
    job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
    years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    submit = st.form_submit_button("Predict Salary")

    if submit:
        gender_enc = encoders['Gender'].transform([gender])[0]
        education_enc = encoders['Education Level'].transform([education])[0]
        job_title_enc = encoders['Job Title'].transform([job_title])[0]
        age_scaled, years_exp_scaled = scaler.transform([[age, years_exp]])[0]
        input_data = pd.DataFrame([[age_scaled, gender_enc, education_enc, job_title_enc, years_exp_scaled]], columns=X.columns)
        predicted_salary = model.predict(input_data)[0]
        st.session_state['predicted_salary'] = predicted_salary

        # --- PDF Download Option ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Employee Salary Prediction Result", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
        pdf.cell(200, 10, txt=f"Education Level: {education}", ln=True)
        pdf.cell(200, 10, txt=f"Job Title: {job_title}", ln=True)
        pdf.cell(200, 10, txt=f"Years of Experience: {years_exp}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt=f"Predicted Salary: ${predicted_salary:,.2f}", ln=True)
        pdf_output = pdf.output(dest='S').encode('latin1')
        st.session_state['pdf_output'] = pdf_output

# Show result and download button outside the form
if st.session_state['predicted_salary'] is not None:
    st.success(f"Predicted Salary: ${st.session_state['predicted_salary']:,.2f}")
    st.download_button(
        label="Download Prediction as PDF",
        data=st.session_state['pdf_output'],
        file_name="salary_prediction.pdf",
        mime="application/pdf"
    )