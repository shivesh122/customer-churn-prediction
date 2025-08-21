# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

# -----------------------------
# UTILS
# -----------------------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target="Churn"):
    df = df.copy()

    # Convert target
    df[target] = df[target].map({"Yes": 1, "No": 0})

    # Drop customerID if exists
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Handle missing values
    df = df.replace(" ", np.nan)
    df = df.dropna()

    # Separate features/target
    X = df.drop(target, axis=1)
    y = df[target]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Build preprocessor
    from sklearn import __version__ as sklearn_version
    if sklearn_version >= "1.2":
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", ohe, categorical),
        ]
    )

    return X, y, preprocessor, numeric, categorical

def train_models(X_train, y_train, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    trained_models = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        trained_models[name] = pipe
    return trained_models

def evaluate_models(models, X_test, y_test):
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })
    return pd.DataFrame(metrics)

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

# -----------------------------
# NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
pages = ["Data Overview", "EDA", "Model Training", "Prediction"]
choice = st.sidebar.radio("Go to", pages)

# -----------------------------
# DATA UPLOAD & LOADING
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = None

st.sidebar.write("### Upload Dataset")
data_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if data_file:
    st.session_state.data = load_data(data_file)

df = st.session_state.data

# -----------------------------
# PAGES
# -----------------------------
if choice == "Data Overview":
    st.title("📊 Customer Churn Dashboard")
    if df is not None:
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write("### Data Summary")
        st.write(df.describe(include="all"))
    else:
        st.info("Upload a dataset to get started.")

elif choice == "EDA":
    st.title("🔎 Exploratory Data Analysis")
    if df is not None:
        st.write("### Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Churn", ax=ax)
        st.pyplot(fig)

        st.write("### Churn by Contract Type")
        if "Contract" in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
            st.pyplot(fig)

        st.write("### Monthly Charges Distribution")
        if "MonthlyCharges" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df, x="MonthlyCharges", hue="Churn", kde=True, ax=ax)
            st.pyplot(fig)

        st.write("### Tenure vs Churn")
        if "tenure" in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Upload a dataset to perform EDA.")

elif choice == "Model Training":
    st.title("🤖 Train ML Models")
    if df is not None:
        X, y, preprocessor, numeric, categorical = preprocess_data(df)

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 100, 42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if st.button("Train Models"):
            models = train_models(X_train, y_train, preprocessor)
            st.session_state.models = models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            st.success("Models trained successfully!")

            metrics_df = evaluate_models(models, X_test, y_test)
            st.write("### Model Performance")
            st.dataframe(metrics_df)

            for name, model in models.items():
                y_pred = model.predict(X_test)
                st.pyplot(plot_confusion(y_test, y_pred, name))
    else:
        st.warning("Upload a dataset to train models.")

elif choice == "Prediction":
    st.title("🔮 Predict Churn for a New Customer")
    if "models" in st.session_state and df is not None:
        X, y, preprocessor, numeric, categorical = preprocess_data(df)

        input_data = {}
        st.write("### Enter Customer Details")

        for col in numeric:
            input_data[col] = st.number_input(f"{col}", value=float(X[col].median()))

        for col in categorical:
            input_data[col] = st.selectbox(f"{col}", options=list(X[col].unique()))

        input_df = pd.DataFrame([input_data])

        model_choice = st.selectbox("Choose Model", list(st.session_state.models.keys()))

        if st.button("Predict"):
            model = st.session_state.models[model_choice]
            prob = model.predict_proba(input_df)[0][1]
            pred = "Yes" if prob > 0.5 else "No"

            st.write(f"### Prediction: {pred}")
            st.write(f"Churn Probability: {prob:.2f}")

            # Save report
            report = input_df.copy()
            report["Prediction"] = pred
            report["Probability"] = prob
            report.to_csv("prediction_report.csv", index=False)
            st.success("Prediction report saved as prediction_report.csv")
    else:
        st.info("Train models first to make predictions.")
