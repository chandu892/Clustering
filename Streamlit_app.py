import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import os

# ================== PATH SETUP ==================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "kmeans_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Mall_Customers.csv")
CLUSTERED_PATH = os.path.join(BASE_DIR, "Clustered_mall_customers.csv")

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Model file not found: {e}")
        return None

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Dataset file not found: {e}")
        return None

@st.cache_data
def load_clustered_data():
    try:
        return pd.read_csv(CLUSTERED_PATH)
    except Exception as e:
        st.error(f"❌ Clustered file not found: {e}")
        return None

model = load_model()
df = load_data()
clustered_df = load_clustered_data()

# ================== CLUSTER INFO ==================
CLUSTER_INFO = {
    0: {"name": "High Value Customers"},
    1: {"name": "Potential Target"},
    2: {"name": "Average Customers"},
    3: {"name": "Loyal Customers"},
    4: {"name": "Budget Conscious"},
}

st.title("🛍️ Mall Customer Clustering Prediction")

# ================== MAIN APP ==================
if model is not None and df is not None and clustered_df is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Customer Input")

        age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 30)
        income = st.slider(
            "Annual Income (k$)",
            int(df["Annual Income (k$)"].min()),
            int(df["Annual Income (k$)"].max()),
            50,
        )
        score = st.slider("Spending Score (1-100)", 1, 100, 50)

    with col2:
        st.subheader("📈 Dataset Info")
        st.metric("Total Customers", len(df))
        st.metric("Avg Age", round(df["Age"].mean(), 1))
        st.metric("Avg Income", round(df["Annual Income (k$)"].mean(), 1))

    # ================== PREDICTION ==================
    if st.button("🚀 Predict Cluster"):
        input_data = pd.DataFrame(
            {"Age": [age], "Annual Income (k$)": [income], "Spending Score (1-100)": [score]}
        )

        scaler = StandardScaler()
        scaler.fit(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])

        scaled_input = scaler.transform(input_data)

        cluster = model.predict(scaled_input)[0]

        st.success(f"✅ Predicted Cluster: {cluster} - {CLUSTER_INFO[cluster]['name']}")

        # ================== VISUALIZATION ==================
        st.subheader("📊 Cluster Visualization")

        fig = px.scatter_3d(
            clustered_df,
            x="Age",
            y="Annual Income (k$)",
            z="Spending Score (1-100)",
            color=clustered_df["Cluster"].astype(str),
            title="3D Customer Clusters",
        )

        fig.add_scatter3d(
            x=[age],
            y=[income],
            z=[score],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Your Input",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ================== PIE CHART ==================
        cluster_counts = clustered_df["Cluster"].value_counts().sort_index()

        fig_pie = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Cluster Distribution",
        )

        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.error("❌ Files missing. Check your project folder.")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("🛍️ Mall Customer Clustering App | KMeans ML Model")
