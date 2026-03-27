import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.title("🏠 California Housing Price Prediction (Ridge Regression)")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df

df = load_data()

st.subheader("📊 Dataset Preview")
st.write(df.head())

# ---------------------------
# Explore
# ---------------------------
st.subheader("📈 Data Description")
st.write(df.describe())

st.subheader("🔍 Missing Values")
st.write(df.isnull().sum())

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
corr = df.corr()

fig, ax = plt.subplots()
cax = ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
fig.colorbar(cax)
st.pyplot(fig)

# ---------------------------
# Prepare Data
# ---------------------------
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
            'Population', 'AveOccup', 'Latitude', 'Longitude']

X = df[features]
y = df['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Model
# ---------------------------
st.subheader("⚙️ Ridge Regression Model")

alpha = st.slider("Select Alpha (Regularization Strength)", 0.01, 10.0, 1.0)

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
# Coefficients
# ---------------------------
st.subheader("📌 Feature Importance (Coefficients)")

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

st.write(coef_df)

# ---------------------------
# Plot Actual vs Predicted
# ---------------------------
st.subheader("📈 Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual Prices")
ax2.set_ylabel("Predicted Prices")
ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)
