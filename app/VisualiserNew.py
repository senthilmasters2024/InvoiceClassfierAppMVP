import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import streamlit as st

def generate_visualization(selected_labels=None):
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

    all_labels = sorted(pd.Series(test_labels).unique().tolist())

    if selected_labels is None:
        st.sidebar.header("📌 Select Top Labels to Visualize")
        selected_labels = st.sidebar.multiselect(
            "Select 3 labels (minimum required)",
            options=all_labels,
            default=all_labels[:3],
            max_selections=3
        )

    if len(selected_labels) < 3:
        st.error("Please select at least 3 distinct labels.")
        return

    filtered_embeddings = []
    filtered_files = []
    filtered_labels = []
    similarities = []
    neighbor_names = []

    for i, label in enumerate(test_labels):
        if label in selected_labels:
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

    color_map = {label: px.colors.qualitative.Plotly[i % 10] for i, label in enumerate(selected_labels)}

    # 2D
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
    st.subheader("📍 2D PCA Visualization")
    st.plotly_chart(fig_2d, use_container_width=True)

    # 3D
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
    st.subheader("📍 3D PCA Visualization")
    st.plotly_chart(fig_3d, use_container_width=True)
