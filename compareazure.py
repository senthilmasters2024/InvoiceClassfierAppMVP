import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

st.set_page_config(layout="wide")
st.title("🔍 KNN vs Azure AI Prediction Comparison Dashboard")

# === Load data ===
BASE_DIR = Path(__file__).resolve().parent
azure_csv = BASE_DIR / "results" / "predictions_enriched_with_top3_similarity.csv"
knn_csv = BASE_DIR / "results" / "predictions.csv"

@st.cache_data
def load_data():
    azure_df = pd.read_csv(azure_csv)
    knn_df = pd.read_csv(knn_csv)
    azure_df["Filename"] = azure_df["Filename"].str.replace(".pdf", "", regex=False).str.strip().str.lower()
    knn_df["Filename"] = knn_df["Filename"].str.replace(".pdf", "", regex=False).str.strip().str.lower()
    return knn_df, azure_df

knn_df, azure_df = load_data()

# === Merge and Compare ===
merged_df = pd.merge(
    knn_df,
    azure_df[["Filename", "PredictedLabel"]],
    on="Filename",
    suffixes=("_KNN", "_Azure")
)

merged_df["LabelMatch"] = (
    merged_df["PredictedLabel_KNN"].str.lower().str.strip() ==
    merged_df["PredictedLabel_Azure"].str.lower().str.strip()
)

st.markdown("### 🧾 Comparison Table")
st.dataframe(merged_df, use_container_width=True)

# === Match/Mismatch Bar Chart ===
match_counts = merged_df["LabelMatch"].value_counts().rename({True: "Match", False: "Mismatch"}).reset_index()
match_counts.columns = ["Type", "Count"]

fig_match = px.bar(match_counts, x="Type", y="Count", title="📊 Prediction Agreement: KNN vs Azure AI")
st.plotly_chart(fig_match, use_container_width=True)

# === Merge similarity columns ===
merged_df["TopSimilarities_KNN"] = knn_df.set_index("Filename").loc[merged_df["Filename"], "TopSimilarities"].values
merged_df["TopSimilarities_Azure"] = azure_df.set_index("Filename").loc[merged_df["Filename"], "TopSimilarities"].values

# Parse and average top similarity scores
def average_similarity(sim_str):
    try:
        return sum(float(s) for s in sim_str.split(",") if s.strip()) / len(sim_str.split(","))
    except:
        return 0.0

merged_df["AvgSimilarity_KNN"] = merged_df["TopSimilarities_KNN"].apply(average_similarity)
merged_df["AvgSimilarity_Azure"] = merged_df["TopSimilarities_Azure"].apply(average_similarity)

# === Display comparison table ===
st.markdown("### 🧾 Label and Similarity Comparison")
st.dataframe(merged_df[[
    "Filename", "PredictedLabel_KNN", "PredictedLabel_Azure",
    "LabelMatch", "AvgSimilarity_KNN", "AvgSimilarity_Azure"
]], use_container_width=True)

# === Confusion Matrix ===
st.markdown("### 🔥 Confusion Matrix")
conf_matrix = pd.crosstab(
    merged_df["PredictedLabel_KNN"],
    merged_df["PredictedLabel_Azure"],
    rownames=["KNN"],
    colnames=["Azure"]
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix: KNN vs Azure")
st.pyplot(fig)

# === Download merged result ===
csv_buffer = BytesIO()
merged_df.to_csv(csv_buffer, index=False)
st.download_button("⬇️ Download Comparison CSV", csv_buffer.getvalue(), "comparison_knn_vs_azure.csv", mime="text/csv")
