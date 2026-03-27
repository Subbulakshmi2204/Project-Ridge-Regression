import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏠 Housing Price Prediction using Ridge Regression")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.frame

df = load_data()

# ---------------------------
# Prepare Data
# ---------------------------
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

X = df[features]
y = df['MedHouseVal']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
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
# Evaluation
# ---------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("📌 Feature Importance")

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

st.bar_chart(coef_df.set_index('Feature'))

# ---------------------------
# Improved Visualization
# ---------------------------
st.subheader("📈 Actual vs Predicted (Color Differentiation)")

fig, ax = plt.subplots()

# Scatter plot with color difference
scatter = ax.scatter(
    y_test,
    y_pred,
    c=(y_test - y_pred),   # color based on error
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

# Color bar for error visualization
cbar = plt.colorbar(scatter)
cbar.set_label("Prediction Error (Actual - Predicted)")

st.pyplot(fig)
