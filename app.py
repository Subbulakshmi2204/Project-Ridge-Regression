import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Housing Price Prediction", layout="wide")

st.title("🏠 Housing Price Prediction using Ridge Regression")

# ---------------------------
# Upload Dataset
# ---------------------------
st.sidebar.header("📂 Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ---------------------------
    # Clean Column Names
    # ---------------------------
    df.columns = df.columns.str.strip().str.lower()

    # ---------------------------
    # Expected Columns (Kaggle Dataset)
    # ---------------------------
    features = [
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income', 'ocean_proximity'
    ]

    target = 'median_house_value'

    # ---------------------------
    # Check Missing Columns
    # ---------------------------
    missing_cols = [col for col in features + [target] if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.write("📌 Available columns:", df.columns.tolist())
        st.stop()

    # ---------------------------
    # Select Required Columns
    # ---------------------------
    df = df[features + [target]]

    # ---------------------------
    # Handle Missing Values
    # ---------------------------
    df = df.dropna()

    # ---------------------------
    # Encode Categorical Feature
    # ---------------------------
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    # ---------------------------
    # Split Data
    # ---------------------------
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Feature Scaling
    # ---------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # Model Section
    # ---------------------------
    st.sidebar.header("⚙️ Model Settings")
    alpha = st.sidebar.slider("Alpha (Regularization)", 0.01, 10.0, 1.0)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    # ---------------------------
    # Prediction
    # ---------------------------
    y_pred = model.predict(X_test_scaled)

    # ---------------------------
    # Evaluation
    # ---------------------------
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📉 Mean Squared Error", f"{mse:.4f}")

    with col2:
        st.metric("📊 R² Score", f"{r2:.4f}")

    # ---------------------------
    # Feature Importance
    # ---------------------------
    st.subheader("📌 Feature Importance")

    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    st.bar_chart(coef_df.set_index('Feature'))

    # ---------------------------
    # Visualization
    # ---------------------------
    st.subheader("📈 Actual vs Predicted (Error Visualization)")

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        y_test,
        y_pred,
        c=(y_test - y_pred),
        cmap='coolwarm',
        alpha=0.7
    )

    # Perfect prediction line
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle='--'
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted Prices")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Prediction Error (Actual - Predicted)")

    st.pyplot(fig)

else:
    st.info("👆 Please upload your California Housing dataset CSV to begin.")
