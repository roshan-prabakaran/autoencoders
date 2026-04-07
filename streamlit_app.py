import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# -----------------------------
# Title
# -----------------------------
st.title("🔍 Autoencoder vs PCA - Anomaly Detection")

st.write("Compare Autoencoder and PCA for detecting anomalies")

# -----------------------------
# Generate Dataset
# -----------------------------
st.sidebar.header("Dataset Settings")

n_samples = st.sidebar.slider("Number of samples", 500, 2000, 1000)
n_features = st.sidebar.slider("Number of features", 5, 50, 20)
anomaly_ratio = st.sidebar.slider("Anomaly Ratio (%)", 1, 20, 10)

# Normal data
X = np.random.normal(0, 1, (n_samples, n_features))

# Inject anomalies
n_anomalies = int(n_samples * anomaly_ratio / 100)
X[:n_anomalies] += np.random.normal(5, 1, (n_anomalies, n_features))

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train = X_scaled[int(0.2*n_samples):]
X_test = X_scaled[:int(0.2*n_samples)]

# -----------------------------
# AUTOENCODER
# -----------------------------
st.subheader("🧠 Autoencoder")

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(10, activation='relu')(input_layer)
encoded = Dense(5, activation='relu')(encoded)

decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=32,
                verbose=0)

# Reconstruction error
reconstructions = autoencoder.predict(X_test, verbose=0)
ae_mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# -----------------------------
# PCA
# -----------------------------
st.subheader("📊 PCA")

pca = PCA(n_components=5)
pca.fit(X_train)

X_pca = pca.transform(X_test)
X_pca_recon = pca.inverse_transform(X_pca)

pca_mse = np.mean(np.power(X_test - X_pca_recon, 2), axis=1)

# -----------------------------
# Threshold
# -----------------------------
st.sidebar.header("Threshold Settings")

threshold_percentile = st.sidebar.slider("Threshold Percentile", 80, 99, 95)

ae_threshold = np.percentile(ae_mse, threshold_percentile)
pca_threshold = np.percentile(pca_mse, threshold_percentile)

ae_anomalies = ae_mse > ae_threshold
pca_anomalies = pca_mse > pca_threshold

# -----------------------------
# Results
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.write("### Autoencoder Results")
    st.write(f"Anomalies detected: {np.sum(ae_anomalies)}")

with col2:
    st.write("### PCA Results")
    st.write(f"Anomalies detected: {np.sum(pca_anomalies)}")

# -----------------------------
# Visualization
# -----------------------------
st.subheader("📈 Reconstruction Error Comparison")

fig, ax = plt.subplots()

ax.hist(ae_mse, bins=50, alpha=0.6, label='Autoencoder')
ax.hist(pca_mse, bins=50, alpha=0.6, label='PCA')

ax.axvline(ae_threshold, linestyle='dashed', label='AE Threshold')
ax.axvline(pca_threshold, linestyle='dashed', label='PCA Threshold')

ax.legend()
ax.set_title("Reconstruction Error Distribution")

st.pyplot(fig)

# -----------------------------
# Scatter Plot (2D PCA view)
# -----------------------------
st.subheader("📍 Visualization (2D Projection)")

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_test)

fig2, ax2 = plt.subplots()

ax2.scatter(X_2d[:, 0], X_2d[:, 1],
            c=ae_anomalies,
            cmap='coolwarm')

ax2.set_title("Autoencoder Detected Anomalies")

st.pyplot(fig2)

# -----------------------------
# Explanation
# -----------------------------
st.markdown("""
### 📌 Key Insights

- Autoencoder captures **non-linear patterns**
- PCA captures **linear variance only**
- Autoencoder usually detects anomalies better in complex data
""")