import sqlite3
import ast
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# === Configuration ===
DB_PATH = Path("../results/embeddings_store.db")  # <-- 🔧 Update this
OUTPUT_PATH = Path("Generated_PCA_With_Embeddings_Spread.xlsx")
PCA_COMPONENTS = 10

# === Load embeddings from SQLite ===
def load_embeddings_from_sqlite(db_path, is_training):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, label, embedding 
        FROM embeddings 
        WHERE is_training = ?
    """, (int(is_training),))
    rows = cursor.fetchall()
    conn.close()

    filenames, labels, embeddings = [], [], []
    for filename, label, emb_str in rows:
        try:
            vec = ast.literal_eval(emb_str)
            if isinstance(vec, list):
                filenames.append(filename)
                labels.append(label)
                embeddings.append(vec)
        except Exception as e:
            print(f"⚠️ Error parsing embedding for {filename}: {e}")
    return filenames, labels, embeddings

# === Build DataFrame with separate cells for each embedding component ===
def build_spread_embedding_pca_df(filenames, labels, embeddings, n_components=10):
    # PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(embeddings)

    # Original embeddings
    emb_df = pd.DataFrame(embeddings, columns=[f"Embedding_{i}" for i in range(len(embeddings[0]))])
    emb_df.insert(0, "Category", labels)
    emb_df.insert(0, "Filename", filenames)

    # PCA columns
    for i in range(n_components):
        emb_df[f"PCA_{i+1}"] = pca_components[:, i]

    return emb_df

# === Export to Excel ===
def export_embeddings_with_pca():
    train_files, train_labels, train_embs = load_embeddings_from_sqlite(DB_PATH, is_training=True)
    test_files, test_labels, test_embs = load_embeddings_from_sqlite(DB_PATH, is_training=False)

    train_df = build_spread_embedding_pca_df(train_files, train_labels, train_embs, PCA_COMPONENTS)
    test_df = build_spread_embedding_pca_df(test_files, test_labels, test_embs, PCA_COMPONENTS)

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        train_df.to_excel(writer, index=False, sheet_name="Training Data")
        test_df.to_excel(writer, index=False, sheet_name="Test Data")

    print(f"✅ Excel file saved to: {OUTPUT_PATH}")

# === Run script ===
if __name__ == "__main__":
    export_embeddings_with_pca()
