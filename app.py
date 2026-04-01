import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("📊 Navigation")
st.sidebar.info("Customer Segmentation using RFM + KMeans")
st.sidebar.markdown("Built for Data Science Portfolio 🚀")

# ----------------------------
# TITLE
# ----------------------------
st.title("🛒 Customer Segmentation (RFM + KMeans)")
st.write("Segment customers using RFM analysis")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("data/Online Retail.xlsx", engine="openpyxl")
    return df

df = load_data()

# ----------------------------
# DATA PREVIEW
# ----------------------------
with st.expander("🔍 View Raw Data"):
    st.write(df.head())

# ----------------------------
# DATA CLEANING
# ----------------------------
df = df.dropna()
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ----------------------------
# RFM CALCULATION
# ----------------------------
snapshot_date = df['InvoiceDate'].max()

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

st.subheader("📊 RFM Table")
st.write(rfm.head())

# ----------------------------
# SCALING
# ----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# ----------------------------
# ELBOW METHOD
# ----------------------------
st.subheader("📉 Elbow Method (Optimal K)")

wcss = []
K_range = range(2, 9)

for i in K_range:
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(rfm_scaled)
    wcss.append(km.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(K_range, wcss, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

# ----------------------------
# K SELECTION
# ----------------------------
k = st.slider("Select number of clusters (K)", 2, 8, 4)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# ----------------------------
# SILHOUETTE SCORE
# ----------------------------
score = silhouette_score(rfm_scaled, rfm['Cluster'])
st.success(f"Silhouette Score: {score:.2f}")

# ----------------------------
# PCA VISUALIZATION
# ----------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(rfm_scaled)

rfm['PCA1'] = pca_data[:, 0]
rfm['PCA2'] = pca_data[:, 1]

st.subheader("📌 Customer Segments Visualization")

fig2, ax2 = plt.subplots()
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax2)
st.pyplot(fig2)

# ----------------------------
# CLUSTER PROFILE
# ----------------------------
st.subheader("📊 Cluster Profile")

profile = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
st.write(profile)

# ----------------------------
# SEGMENT NAMING (Improved)
# ----------------------------
def segment(row):
    if row['Monetary'] > rfm['Monetary'].quantile(0.75):
        return 'High Value'
    elif row['Frequency'] > rfm['Frequency'].quantile(0.75):
        return 'Loyal'
    elif row['Recency'] > rfm['Recency'].quantile(0.75):
        return 'At Risk'
    else:
        return 'Low Value'

rfm['Segment'] = rfm.apply(segment, axis=1)

# ----------------------------
# SEGMENT DISTRIBUTION
# ----------------------------
st.subheader("📊 Segment Distribution")

fig3, ax3 = plt.subplots()
rfm['Segment'].value_counts().plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# ----------------------------
# FINAL DATA
# ----------------------------
st.subheader("📄 Final Data with Segments")
st.write(rfm.head())

# ----------------------------
# DOWNLOAD BUTTON
# ----------------------------
csv = rfm.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Segmented Data",
    data=csv,
    file_name='customer_segments.csv',
    mime='text/csv',
)

