import os
import json
import sqlite3

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio

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


def generate_pca_knn_visualizations():
    _generate_pca_visualizations(source='knn')

def generate_pca_azure_visualizations():
    _generate_pca_visualizations(source='azure')

def _generate_pca_visualizations(source='knn'):
    BASE_DIR = Path(__file__).resolve().parent.parent
    SQLITE_PATH = BASE_DIR / "results" / "embeddings_store.db"

    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query("SELECT filename, label, embedding, is_training FROM embeddings", conn)
    conn.close()

    result_csv = BASE_DIR / "results" / (
        "predictions.csv" if source == 'knn' else "predictions_enriched_with_top3_similarity.csv"
    )
    results_df = pd.read_csv(result_csv)
    results_df['Filename'] = results_df['Filename'].str.replace('.pdf', '', regex=False).str.lower()

    embeddings, labels, filenames = [], [], []

    for _, row in df.iterrows():
        try:
            vec = json.loads(row["embedding"])
            fname = row["filename"].replace('.pdf', '').lower()
            label = row["label"]

            if row["is_training"]:
                continue

            match = results_df[results_df["Filename"] == fname]
            if match.empty:
                continue
            pred_label = match.iloc[0]["PredictedLabel"]
            embeddings.append(vec)
            labels.append(pred_label)
            filenames.append(fname)
        except:
            continue

    if not embeddings:
        return

    color_palette = px.colors.qualitative.Set2
    unique_labels = list(set(labels))
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

    # 2D PCA
    pca_2d = PCA(n_components=2)
    reduced_2d = pca_2d.fit_transform(np.array(embeddings))
    df_2d = pd.DataFrame({
        "x": reduced_2d[:, 0],
        "y": reduced_2d[:, 1],
        "label": labels,
        "filename": filenames
    })
    fig_2d = px.scatter(df_2d, x="x", y="y", color="label", color_discrete_map=color_map,
                        hover_data=["filename", "label"])
    pio.write_html(fig_2d, file=str(BASE_DIR / "results" / f"{source}_pca_2d.html"))

    # 3D PCA
    pca_3d = PCA(n_components=3)
    reduced_3d = pca_3d.fit_transform(np.array(embeddings))
    df_3d = pd.DataFrame({
        "x": reduced_3d[:, 0],
        "y": reduced_3d[:, 1],
        "z": reduced_3d[:, 2],
        "label": labels,
        "filename": filenames
    })
    fig_3d = px.scatter_3d(df_3d, x="x", y="x", z="z", color="label", color_discrete_map=color_map,
                           hover_data=["filename", "label"])
    pio.write_html(fig_3d, file=str(BASE_DIR / "results" / f"{source}_pca_3d.html"))

import umap
from sklearn.manifold import TSNE

def generate_pca_knn_visualizations(labels_filter=None, reducer='PCA'):
    _generate_pca_visualizations(source='knn', labels_filter=labels_filter, reducer=reducer)

def generate_pca_azure_visualizations(labels_filter=None, reducer='PCA'):
    _generate_pca_visualizations(source='azure', labels_filter=labels_filter, reducer=reducer)

def _generate_pca_visualizations(source='knn', labels_filter=None, reducer='PCA'):
    BASE_DIR = Path(__file__).resolve().parent.parent
    SQLITE_PATH = BASE_DIR / "results" / "embeddings_store.db"

    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query("SELECT filename, label, embedding, is_training FROM embeddings", conn)
    conn.close()

    result_csv = BASE_DIR / "results" / (
        "predictions.csv" if source == 'knn' else "predictions_enriched_with_top3_similarity.csv"
    )
    results_df = pd.read_csv(result_csv)
    results_df['Filename'] = results_df['Filename'].str.replace('.pdf', '', regex=False).str.lower()

    embeddings, labels, filenames, shapes = [], [], [], []

    for _, row in df.iterrows():
        try:
            vec = json.loads(row["embedding"])
            fname = row["filename"].replace('.pdf', '').lower()
            label = row["label"]
            if row["is_training"]:
                if labels_filter and label not in labels_filter:
                    continue
                embeddings.append(vec)
                labels.append(label)
                filenames.append(fname)
                shapes.append("train")
            else:
                match = results_df[results_df["Filename"] == fname]
                if match.empty:
                    continue
                pred_label = match.iloc[0]["PredictedLabel"]
                if labels_filter and pred_label not in labels_filter:
                    continue
                embeddings.append(vec)
                labels.append(pred_label)
                filenames.append(fname)
                shapes.append("test")
        except:
            continue


    # Override neighbors/similarities from Azure CSV if source is azure
    if source == 'azure':
        filename_to_neighbors = dict(zip(results_df["Filename"], results_df["TopNeighbors"]))
        filename_to_similarities = dict(zip(results_df["Filename"], results_df["TopSimilarities"]))
        neighbors = [filename_to_neighbors.get(fname, "") for fname in filenames]
        similarities = [filename_to_similarities.get(fname, "") for fname in filenames]


    if len(set(labels)) < 3:
        return

    color_palette = px.colors.qualitative.Set3
    unique_labels = list(set(labels))
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

    # Choose dimensionality reducer
    if reducer.lower() == "umap":
        reducer_model = umap.UMAP(n_components=2)
    elif reducer.lower() == "t-sne":
        reducer_model = TSNE(n_components=2)
    else:
        reducer_model = PCA(n_components=2)

    reduced_2d = reducer_model.fit_transform(np.array(embeddings))
    df_2d = pd.DataFrame({
        "x": reduced_2d[:, 0],
        "y": reduced_2d[:, 1],
        "label": labels,
        "filename": filenames,
        "shape": shapes
    })
    fig_2d = px.scatter(df_2d, x="x", y="y", color="label", symbol="shape", color_discrete_map=color_map,
                        hover_data=["filename", "label", "shape"])
    output_name = f"{source.lower()}_{reducer.lower()}_2d.html"
    pio.write_html(fig_2d, file=str(BASE_DIR / "results" / output_name))


def _generate_pca_visualizations(source='knn', labels_filter=None, reducer='PCA'):
    BASE_DIR = Path(__file__).resolve().parent.parent
    SQLITE_PATH = BASE_DIR / "results" / "embeddings_store.db"

    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query("SELECT filename, label, embedding, is_training, top_neighbors, top_similarities FROM embeddings", conn)
    conn.close()

    result_csv = BASE_DIR / "results" / (
        "predictions.csv" if source == 'knn' else "predictions_enriched_with_top3_similarity.csv"
    )
    results_df = pd.read_csv(result_csv)
    results_df['Filename'] = results_df['Filename'].str.replace('.pdf', '', regex=False).str.lower()

    embeddings, labels, filenames, shapes = [], [], [], []
    neighbors, similarities = [], []

    for _, row in df.iterrows():
        try:
            vec = json.loads(row["embedding"])
            fname = row["filename"].replace('.pdf', '').lower()
            label = row["label"]
            if row["is_training"]:
                if labels_filter and label not in labels_filter:
                    continue
                embeddings.append(vec)
                labels.append(label)
                filenames.append(fname)
                shapes.append("train")
                neighbors.append(row["top_neighbors"])
                similarities.append(row["top_similarities"])
            else:
                match = results_df[results_df["Filename"] == fname]
                if match.empty:
                    continue
                pred_label = match.iloc[0]["PredictedLabel"]
                if labels_filter and pred_label not in labels_filter:
                    continue
                embeddings.append(vec)
                labels.append(pred_label)
                filenames.append(fname)
                shapes.append("test")
                neighbors.append(row["top_neighbors"])
                similarities.append(row["top_similarities"])
        except:
            continue


    # Override neighbors/similarities from Azure CSV if source is azure
    if source == 'azure':
        filename_to_neighbors = dict(zip(results_df["Filename"], results_df["TopNeighbors"]))
        filename_to_similarities = dict(zip(results_df["Filename"], results_df["TopSimilarities"]))
        neighbors = [filename_to_neighbors.get(fname, "") for fname in filenames]
        similarities = [filename_to_similarities.get(fname, "") for fname in filenames]


    if len(set(labels)) < 3:
        return

    unique_labels = list(set(labels))
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(unique_labels)}

    reducer_model = {
        "pca": PCA(n_components=2),
        "umap": umap.UMAP(n_components=2),
        "t-sne": TSNE(n_components=2)
    }.get(reducer.lower(), PCA(n_components=2))

    reduced_2d = reducer_model.fit_transform(np.array(embeddings))
    df_2d = pd.DataFrame({
        "x": reduced_2d[:, 0],
        "y": reduced_2d[:, 1],
        "label": labels,
        "filename": filenames,
        "shape": shapes,
        "neighbors": neighbors,
        "similarities": similarities
    })
    fig_2d = px.scatter(df_2d, x="x", y="y", color="label", symbol="shape", color_discrete_map=color_map,
                        hover_data=["filename", "label", "shape", "neighbors", "similarities"])
    pio.write_html(fig_2d, file=str(BASE_DIR / "results" / f"{source.lower()}_{reducer.lower()}_2d.html"))

    reducer_3d_model = {
        "pca": PCA(n_components=3),
        "umap": umap.UMAP(n_components=3),
        "t-sne": TSNE(n_components=3)
    }.get(reducer.lower(), PCA(n_components=3))

    reduced_3d = reducer_3d_model.fit_transform(np.array(embeddings))
    df_3d = pd.DataFrame({
        "x": reduced_3d[:, 0],
        "y": reduced_3d[:, 1],
        "z": reduced_3d[:, 2],
        "label": labels,
        "filename": filenames,
        "shape": shapes,
        "neighbors": neighbors,
        "similarities": similarities
    })
    fig_3d = px.scatter_3d(df_3d, x="x", y="y", z="z", color="label", symbol="shape", color_discrete_map=color_map,
                           hover_data=["filename", "label", "shape", "neighbors", "similarities"])
    pio.write_html(fig_3d, file=str(BASE_DIR / "results" / f"{source.lower()}_{reducer.lower()}_3d.html"))