import os
import json
import pickle
import numpy as np
import sqlite3
import shutil
import pandas as pd
import csv
import ast
from pathlib import Path
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

from utils.pdf_text import extract_text_from_pdf

# OpenAI initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🔐 OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
EMBED_DIR = BASE_DIR / "embeddings"
MODEL_PATH = BASE_DIR / "models" / "knn_model.pkl"
RESULT_PATH = BASE_DIR / "results" / "predictions.csv"
SIM_MATRIX_PATH = BASE_DIR / "results" / "similarity_matrix.csv"
CLASSIFIED_DIR = BASE_DIR / "classified"
CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = BASE_DIR / "results" / "embeddings_store.db"
MAX_CHARS = 12000

# === SQLite Setup ===
def initialize_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label TEXT,
            embedding TEXT NOT NULL,
            is_training BOOLEAN NOT NULL,
            pca_x REAL,
            pca_y REAL,
            top_neighbors TEXT,
            top_similarities TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_embedding_to_sqlite(filename, label, embedding, is_training, top_neighbors=None, top_similarities=None):
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO embeddings (filename, label, embedding, is_training, top_neighbors, top_similarities)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (filename, label, json.dumps(embedding), int(is_training), top_neighbors, top_similarities)
    )
    conn.commit()
    conn.close()

def update_pca_coordinates(pca_results, ids):
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    for i, (x, y) in enumerate(pca_results):
        cursor.execute("UPDATE embeddings SET pca_x = ?, pca_y = ? WHERE id = ?", (x, y, ids[i]))
    conn.commit()
    conn.close()

# === Embedding Utilities ===
def chunk_text(text, max_chars=MAX_CHARS):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def get_avg_embedding_from_chunks(text):
    chunks = chunk_text(text)
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-large"
    )
    embeddings = [np.array(r.embedding) for r in response.data]
    avg_vector = np.mean(embeddings, axis=0)
    return avg_vector.tolist()

def embed_and_cache(filepath, label=None):
    filename = filepath.name
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding, label FROM embeddings WHERE filename = ?", (filename,))
    row = cursor.fetchone()
    conn.close()

    if row:
        embedding = json.loads(row[0])
        existing_label = row[1]
        return embedding, existing_label or label or "unknown"

    safe_name = filepath.stem.replace(" ", "_").replace("/", "_")
    embed_file = EMBED_DIR / f"{safe_name}.json"
    if embed_file.exists():
        with open(embed_file, "r") as f:
            obj = json.load(f)
            return obj["embedding"], obj.get("label", "unknown")

    text = extract_text_from_pdf(filepath)
    embedding = get_avg_embedding_from_chunks(text)
    with open(embed_file, "w") as f:
        json.dump({"embedding": embedding, "label": label or "unknown"}, f, indent=2)

    return embedding, label or "unknown"

# === KNN Training ===
def train_knn_model():
    X, y, filenames = [], [], []
    initialize_sqlite()

    for folder in (UPLOAD_DIR / "train").glob("*"):
        label = folder.name
        for file in folder.glob("*.pdf"):
            embedding, _ = embed_and_cache(file, label)
            insert_embedding_to_sqlite(file.name, label, embedding, True, None, None)
            X.append(embedding)
            y.append(label)
            filenames.append(file.name)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    model._fit_X_filenames = filenames

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

# === Classification and Post-Processing ===
def classify_test_documents(model):
    X_test, filenames, embeddings = [], [], []

    for file in (UPLOAD_DIR / "test").glob("*.pdf"):
        embedding, _ = embed_and_cache(file)
        X_test.append(embedding)
        embeddings.append(embedding)
        filenames.append(file.name)

    predicted_labels = model.predict(X_test)

    with open(RESULT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Filename", "PredictedLabel", "TopNeighbors", "TopSimilarities"])
        for i, embedding in enumerate(X_test):
            distances, indices = model.kneighbors([embedding], n_neighbors=3)
            top_neighbor_filenames = [model._fit_X_filenames[idx] for idx in indices[0]]
            top_similarities = [f"{1 - distances[0][j]:.4f}" for j in range(len(indices[0]))]

            neighbor_info = "; ".join(top_neighbor_filenames)
            similarity_info = "; ".join(top_similarities)

            w.writerow([filenames[i], predicted_labels[i], neighbor_info, similarity_info])
            insert_embedding_to_sqlite(filenames[i], predicted_labels[i], embedding, False, neighbor_info, similarity_info)

    for i, file in enumerate((UPLOAD_DIR / "test").glob("*.pdf")):
        predicted_label = str(predicted_labels[i])
        target_dir = CLASSIFIED_DIR / predicted_label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, target_dir / file.name)

    similarity_matrix = cosine_similarity(X_test)
    with open(SIM_MATRIX_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + filenames)
        for i, row in enumerate(similarity_matrix):
            w.writerow([filenames[i]] + [f"{sim:.4f}" for sim in row])

    conn = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql_query("SELECT id, embedding FROM embeddings", conn)
    conn.close()

    ids, vectors = [], []
    for idx, emb_str in zip(df["id"], df["embedding"]):
        try:
            vec = ast.literal_eval(emb_str)
            if isinstance(vec, list) and len(vec) > 100:
                ids.append(idx)
                vectors.append(vec)
        except:
            continue

    if vectors:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(vectors))
        update_pca_coordinates(reduced, ids)

    return list(zip(filenames, predicted_labels)), predicted_labels
