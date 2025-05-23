# app/visualizer.py
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio

def generate_visualization():
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

    top_3_labels = pd.Series(test_labels).value_counts().head(3).index.tolist()
    if len(top_3_labels) < 3:
        raise ValueError("At least 3 unique predicted labels are required.")

    filtered_embeddings = []
    filtered_files = []
    filtered_labels = []
    similarities = []
    neighbor_names = []

    for i, label in enumerate(test_labels):
        if label in top_3_labels:
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

    color_map = {
        top_3_labels[0]: "#1f77b4",
        top_3_labels[1]: "#ff7f0e",
        top_3_labels[2]: "#2ca02c",
    }

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
    pio.write_html(fig_2d, file=str(BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"))

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
    pio.write_html(fig_3d, file=str(BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"))
# app/visualizer.py
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio

def generate_visualization():
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

    top_3_labels = pd.Series(test_labels).value_counts().head(3).index.tolist()
    if len(top_3_labels) < 3:
        raise ValueError("At least 3 unique predicted labels are required.")

    filtered_embeddings = []
    filtered_files = []
    filtered_labels = []
    similarities = []
    neighbor_names = []

    for i, label in enumerate(test_labels):
        if label in top_3_labels:
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

    color_map = {
        top_3_labels[0]: "#1f77b4",
        top_3_labels[1]: "#ff7f0e",
        top_3_labels[2]: "#2ca02c",
    }

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
    pio.write_html(fig_2d, file=str(BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"))

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
    pio.write_html(fig_3d, file=str(BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"))
