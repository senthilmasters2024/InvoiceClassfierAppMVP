# app.py (Integrated)

import io
import os
import sys
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components
import zipfile
from visualize_classification import generate_visualization
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio
# Internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.file_handler as file_handler
import classifier as classifier
import visualize_classification
from azuresearchclienttrail import classify_invoices_with_vector_search

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_CSV = BASE_DIR / "results" / "predictions.csv"
AZURE_CSV = BASE_DIR / "results" / "predictions_enriched_with_top3_similarity.csv"
INVOICE_DIR = BASE_DIR / "uploads" / "test"
PLOT_2D_HTML = BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"
PLOT_3D_HTML = BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"
CLASSIFIED_DIR = BASE_DIR / "classified"

st.set_page_config(page_title="Unified Invoice App", layout="wide")

tab1, tab2 = st.tabs(["📥 Upload & Classify", "📊 Dashboard"])
@st.cache_data
def load_invoice_categories():
    config_path = Path(__file__).resolve().parent.parent / "config" / "categories.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            return config.get("invoice_categories", [])
    return []
# === Tab 1: Upload & Classify ===
with tab1:
    st.header("📂 Upload Training Data")
    label = st.selectbox("Select Label/Category", ["healthcare", "capitalincome", "craftsman", "energy", "other"])
    train_files = st.file_uploader("Upload .pdf or .txt files for training", type=["pdf", "txt"], accept_multiple_files=True, key="train_upload")

    if train_files:
        file_handler.save_training_files(label, train_files)
        st.success(f"✅ Uploaded {len(train_files)} training files to '{label}' category.")

    st.header("🧾 Upload Invoices to Classify")
    test_files = st.file_uploader("Upload invoice documents (.pdf only)", type=["pdf"], accept_multiple_files=True, key="test_upload")

    if test_files:
        file_handler.save_test_files(test_files)
        st.success(f"✅ Uploaded {len(test_files)} invoices for classification.")

    st.header("📁 Folder Preview")
    if st.button("Show Uploaded Data"):
        file_handler.show_uploaded_files()

    if st.button("🧠 Train & Classify using KNN"):
        with st.spinner("Generating embeddings and training model..."):
            model = classifier.train_knn_model()
            results, predicted_labels = classifier.classify_test_documents(model)
            classifier.perform_pca()
        st.success("✅ Classification complete!")
        st.write("**Results:**")
        for fname, label in results:
            st.markdown(f"- `{fname}` → **{label}**")

    if st.button("📈 Generate Visualizations"):
        with st.spinner("Generating visualizations..."):
            visualize_classification.generate_visualization()
        st.success("✅ Visualizations created. Check the Dashboard tab.")

    if st.button("☁️ Run Azure Vector Search Classification"):
        with st.spinner("Classifying with Azure AI Search..."):
            classify_invoices_with_vector_search()
        st.success("✅ Azure classification complete! Check the Dashboard tab.")

# === Tab 2: Dashboard ===
with tab2:
    st.title("📊 Invoice Classification Dashboard")

    @st.cache_data
    def load_data():
        knn_df = pd.read_csv(RESULT_CSV)
        azure_df = pd.read_csv(AZURE_CSV)
        for df in [knn_df, azure_df]:
            df["Filename"] = df["Filename"].str.replace(".pdf", "", regex=False).str.strip().str.lower()
        return knn_df, azure_df

    knn_df, azure_df = load_data()
    all_files = sorted(set(knn_df["Filename"]).union(set(azure_df["Filename"])))

    selected_invoice = st.sidebar.selectbox("📄 Select Invoice Filename", all_files)

    st.sidebar.markdown("---")
    if selected_invoice:
        knn_row = knn_df[knn_df["Filename"] == selected_invoice]
        azure_row = azure_df[azure_df["Filename"] == selected_invoice]

        st.sidebar.markdown("### 🔎 Prediction Overview")
        if not knn_row.empty:
            st.sidebar.markdown(f"**KNN Label:** `{knn_row.iloc[0]['PredictedLabel']}`")
            st.sidebar.markdown(f"**KNN Similarities:** `{knn_row.iloc[0]['TopSimilarities']}`")
            st.sidebar.markdown(f"**KNN Neighbors:** `{knn_row.iloc[0]['TopNeighbors']}`")
        if not azure_row.empty:
            st.sidebar.markdown(f"**Azure Label:** `{azure_row.iloc[0]['PredictedLabel']}`")
            st.sidebar.markdown(f"**Azure Similarities:** `{azure_row.iloc[0]['TopSimilarities']}`")
            st.sidebar.markdown(f"**Azure Neighbors:** `{azure_row.iloc[0]['TopNeighbors']}`")

    # Process & merge for tables
    knn_df["KNN_TopNeighbors"] = knn_df["TopNeighbors"].astype(str).str.split(",").apply(lambda x: ", ".join(x[:3]))
    azure_df["Azure_TopNeighbors"] = azure_df["TopNeighbors"].astype(str).str.split(",").apply(lambda x: ", ".join(x[:3]))
    knn_df["KNN_TopSimilarities"] = knn_df["TopSimilarities"].str.replace(";", ",")
    azure_df["Azure_TopSimilarities"] = azure_df["TopSimilarities"].str.replace(";", ",")

    merged = pd.merge(
        knn_df[["Filename", "PredictedLabel", "KNN_TopSimilarities", "KNN_TopNeighbors"]].rename(columns={"PredictedLabel": "PredictedLabel_KNN"}),
        azure_df[["Filename", "PredictedLabel", "Azure_TopSimilarities", "Azure_TopNeighbors"]].rename(columns={"PredictedLabel": "PredictedLabel_Azure"}),
        on="Filename", how="inner"
    )

    st.subheader("📄 Invoice Prediction Summary")
    st.dataframe(merged, use_container_width=True)

    st.subheader("📊 Visual Embedding Comparison")


    def generate_visualization(labels_filter=None):
        BASE_DIR = Path(__file__).resolve().parent.parent
        EMBED_DIR = BASE_DIR / "embeddings"
        RESULT_CSV = BASE_DIR / "results" / "predictions.csv"

        df = pd.read_csv(RESULT_CSV)
        df['normalized'] = df['Filename'].str.replace('.pdf', '', regex=False)

        train_embeddings, train_labels, train_files = [], [], []
        test_embeddings, test_labels, test_files = [], [], []

        for file in EMBED_DIR.glob("*.json"):
            if "Similarity" in file.name:
                continue
            with open(file, "r") as f:
                data = json.load(f)
                vector = data["embedding"]
                label = data.get("label", "unknown").lower()
                filename = file.stem.replace(" ", "_").replace("-", "_")
                row = df[df["normalized"].str.replace(" ", "_").str.replace("-", "_") == filename]
                if row.empty:
                    train_embeddings.append(vector)
                    train_labels.append(label)
                    train_files.append(filename)
                else:
                    test_embeddings.append(vector)
                    test_labels.append(row["PredictedLabel"].values[0])
                    test_files.append(filename)

        if labels_filter is None:
            labels_filter = pd.Series(test_labels).value_counts().head(3).index.tolist()

        if len(labels_filter) < 3:
            raise ValueError("At least 3 unique predicted labels are required.")

        filtered_embeddings = []
        filtered_files = []
        filtered_labels = []
        similarities = []
        neighbor_names = []

        for i, label in enumerate(test_labels):
            if label in labels_filter:
                test_vec = np.array(test_embeddings[i]).reshape(1, -1)
                train_vecs = np.array(train_embeddings)
                sim_scores = cosine_similarity(test_vec, train_vecs)[0]
                max_idx = np.argmax(sim_scores)
                max_sim = sim_scores[max_idx]
                nearest_train_filename = train_files[max_idx] if max_idx < len(train_files) else "unknown"
                similarities.append(round(max_sim, 4))
                neighbor_names.append(nearest_train_filename)
                filtered_embeddings.append(test_embeddings[i])
                filtered_files.append(test_files[i])
                filtered_labels.append(label)

        color_palette = px.colors.qualitative.Set1
        color_map = {label: color_palette[i] for i, label in enumerate(labels_filter)}

        pca_2d = PCA(n_components=2)
        reduced_2d = pca_2d.fit_transform(filtered_embeddings)
        df_2d = pd.DataFrame({
            "x": reduced_2d[:, 0],
            "y": reduced_2d[:, 1],
            "filename": filtered_files,
            "label": filtered_labels,
            "similarity": similarities,
            "nearest_train_file": neighbor_names
        })
        fig_2d = px.scatter(df_2d, x="x", y="y", color="label", color_discrete_map=color_map,
                            hover_data=["filename", "label", "similarity", "nearest_train_file"])
        name_2d = f"pca_2d_{'_'.join(labels_filter)}.html"
        pio.write_html(fig_2d, file=str(BASE_DIR / "results" / name_2d))

        pca_3d = PCA(n_components=3)
        reduced_3d = pca_3d.fit_transform(filtered_embeddings)
        df_3d = pd.DataFrame({
            "x": reduced_3d[:, 0],
            "y": reduced_3d[:, 1],
            "z": reduced_3d[:, 2],
            "filename": filtered_files,
            "label": filtered_labels,
            "similarity": similarities,
            "nearest_train_file": neighbor_names
        })
        fig_3d = px.scatter_3d(df_3d, x="x", y="y", z="z", color="label", color_discrete_map=color_map,
                               hover_data=["filename", "label", "similarity", "nearest_train_file"])
        name_3d = f"pca_3d_{'_'.join(labels_filter)}.html"
        pio.write_html(fig_3d, file=str(BASE_DIR / "results" / name_3d))
with st.expander("KNN PCA Visualization (2D + 3D)"):
    try:
        with open(BASE_DIR / "results" / "knn_pca_2d.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=500, scrolling=True)
        with open(BASE_DIR / "results" / "knn_pca_3d.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600, scrolling=True)
    except FileNotFoundError:
        st.warning("KNN PCA plots not generated yet.")

with st.expander("Azure PCA Visualization (2D + 3D)"):
    try:
        with open(BASE_DIR / "results" / "azure_pca_2d.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=500, scrolling=True)
        with open(BASE_DIR / "results" / "azure_pca_3d.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600, scrolling=True)
    except FileNotFoundError:
        st.warning("Azure PCA plots not generated yet.")

# Add in Tab 1 section for triggers:

if st.button("📈 Generate KNN PCA Plots"):
    from visualize_classification import generate_pca_knn_visualizations
    generate_pca_knn_visualizations()
    st.success("✅ KNN PCA plots generated!")

if st.button("📈 Generate Azure PCA Plots"):
    from visualize_classification import generate_pca_azure_visualizations
    generate_pca_azure_visualizations()
    st.success("✅ Azure PCA plots generated!")

# Add in Tab 1 for plot generation options:
with st.expander("🔧 Visualization Settings"):
    selected_labels = st.multiselect("Select up to 3 categories to visualize", options=sorted(knn_df['PredictedLabel'].unique()), default=sorted(knn_df['PredictedLabel'].unique())[:3])
    reducer = st.selectbox("Dimensionality Reduction Method", ["PCA", "UMAP", "t-SNE"])

if st.button("📊 Generate Advanced KNN Plots"):
    from visualize_classification import generate_pca_knn_visualizations
    generate_pca_knn_visualizations(selected_labels, reducer)
    st.success("✅ Advanced KNN plots created!")

if st.button("📊 Generate Advanced Azure Plots"):
    from visualize_classification import generate_pca_azure_visualizations
    generate_pca_azure_visualizations(selected_labels, reducer)
    st.success("✅ Advanced Azure plots created!")