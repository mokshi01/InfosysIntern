# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris ML App", layout="centered")

# Title and description
st.title("Iris Flower Classifier 🌸")
st.write("Predict the species of an Iris flower using a trained Random Forest model.")

# Load model and dataset
MODEL_PATH = "iris_model.joblib"

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]
    target_names = saved["target_names"]
except:
    st.warning("Model not found! Please run train_model.py first.")
    st.stop()

iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Sidebar: Mode selection
mode = st.sidebar.radio("Choose mode:", ["Prediction", "Data Exploration"])

# ------------------ Prediction Mode ------------------
if mode == "Prediction":
    st.subheader("Prediction Mode")
    st.write("Adjust the sliders to input flower features:")

    # Input sliders
    inputs = []
    for feature in feature_names:
        val = st.slider(
            label=feature.capitalize(),
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean()),
            step=0.1,
            help=f"Select {feature}",
        )
        inputs.append(val)

    # Prediction
    if st.button("Predict"):
        prediction = model.predict([inputs])[0]
        prediction_proba = model.predict_proba([inputs])[0]

        st.write(f"**Predicted Species:** {target_names[prediction]}")
        st.write("**Prediction Probabilities:**")
        proba_df = pd.DataFrame([prediction_proba], columns=target_names)
        st.dataframe(proba_df.style.background_gradient(cmap="Blues"))

# ------------------ Data Exploration Mode ------------------
else:
    st.subheader("Data Exploration Mode")
    st.write("Explore the Iris dataset:")

    # Histogram
    feature = st.selectbox("Select feature for histogram:", feature_names)
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=15, color="skyblue", edgecolor="black")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of {feature}")
    st.pyplot(fig)

    # Scatter plot
    st.write("Scatter plot of two features:")
    x_feature = st.selectbox("X-axis feature:", feature_names, index=0)
    y_feature = st.selectbox("Y-axis feature:", feature_names, index=1)
    fig2, ax2 = plt.subplots()
    ax2.scatter(df[x_feature], df[y_feature], c=df["target"], cmap="viridis")
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    ax2.set_title(f"{y_feature} vs {x_feature}")
    st.pyplot(fig2)
