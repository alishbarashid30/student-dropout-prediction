import pandas as pd
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("student_dropout.csv", sep=";")


# =========================
# CREATE BINARY TARGET
# =========================

df["Dropout"] = (df["Target"] == "Dropout").astype(int)

X = df.drop(["Target", "Dropout"], axis=1)
y = df["Dropout"]


# =========================
# ENCODE CATEGORICAL DATA
# =========================

X = pd.get_dummies(X, drop_first=True)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# =========================
# PREDICTION FUNCTION
# =========================

def predict_student_risk(
    age,
    admission_grade,
    previous_grade,
    enrolled_1st,
    approved_1st,
    grade_1st,
    enrolled_2nd,
    approved_2nd,
    grade_2nd,
    tuition,
    debtor,
    scholarship
):

    student_data = {
        "Marital status": 1,
        "Application mode": 17,
        "Application order": 1,
        "Course": 171,
        "Daytime/evening attendance\t": 1,
        "Previous qualification": 1,
        "Previous qualification (grade)": previous_grade,
        "Nacionality": 1,
        "Mother's qualification": 13,
        "Father's qualification": 10,
        "Mother's occupation": 6,
        "Father's occupation": 8,
        "Admission grade": admission_grade,
        "Displaced": 1,
        "Educational special needs": 0,
        "Debtor": int(debtor),
        "Tuition fees up to date": int(tuition),
        "Gender": 1,
        "Scholarship holder": int(scholarship),
        "Age at enrollment": age,
        "International": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": enrolled_1st,
        "Curricular units 1st sem (evaluations)": enrolled_1st,
        "Curricular units 1st sem (approved)": approved_1st,
        "Curricular units 1st sem (grade)": grade_1st,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": enrolled_2nd,
        "Curricular units 2nd sem (evaluations)": enrolled_2nd,
        "Curricular units 2nd sem (approved)": approved_2nd,
        "Curricular units 2nd sem (grade)": grade_2nd,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10,
        "Inflation rate": 2,
        "GDP": 1.5
    }

    input_df = pd.DataFrame([student_data])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=X.columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return f"{probability * 100:.2f}%", risk


# =========================
# GRADIO APPLICATION
# =========================

interface = gr.Interface(
    fn=predict_student_risk,

    inputs=[
        gr.Number(label="Age at Enrollment", value=19),
        gr.Number(label="Admission Grade", value=130),
        gr.Number(label="Previous Qualification Grade", value=120),

        gr.Number(label="1st Semester Enrolled", value=6),
        gr.Number(label="1st Semester Approved", value=5),
        gr.Number(label="1st Semester Grade", value=13),

        gr.Number(label="2nd Semester Enrolled", value=6),
        gr.Number(label="2nd Semester Approved", value=5),
        gr.Number(label="2nd Semester Grade", value=13),

        gr.Number(label="Tuition Fees Up to Date", value=1),
        gr.Number(label="Debtor", value=0),
        gr.Number(label="Scholarship Holder", value=0)
    ],

    outputs=[
        gr.Textbox(label="Dropout Probability"),
        gr.Textbox(label="Risk Category")
    ],

    title="Student Dropout Risk Prediction",

    description=(
        "Enter student information to predict "
        "dropout probability and risk level."
    )
)


interface.launch()
