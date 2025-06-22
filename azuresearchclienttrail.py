import ast
import sqlite3
import sys
import json
import requests
import pandas as pd
from pathlib import Path
import re

# === Setup ===
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "results" / "embeddings_store.db"
RESULT_CSV = BASE_DIR / "results" / "top3_predictions_with_scores.csv"
ENRICHED_CSV = BASE_DIR / "results" / "predictions_enriched_with_top3_similarity.csv"
RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

print("Python EXE:", sys.executable)

# Azure AI Search Config
api_key = ""
search_service = "classificationaisearch1"
index_name = "classificationrag-1748959705585"
vector_field = "text_vector"
semantic_config = "classificationrag-1748959705585-semantic-configuration"
api_version = "2025-05-01-preview"

# === Get New Embeddings (Unprocessed) ===
def get_new_invoice_embeddings(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, embedding 
        FROM embeddings 
        WHERE is_training = 0 AND (is_searched IS NULL OR is_searched = 0)
    """)
    rows = cursor.fetchall()
    conn.close()

    data = []
    for filename, emb_str in rows:
        try:
            embedding = ast.literal_eval(emb_str)
            if isinstance(embedding, list):
                embedding = [float(x) for x in embedding]
                data.append((filename, embedding))
        except Exception as e:
            print(f"⚠️ Could not parse embedding for {filename}: {e}")
    return data

# === Vector Search Classification ===
def classify_invoices_with_vector_search():
    invoice_data = get_new_invoice_embeddings()
    endpoint = f"https://{search_service}.search.windows.net/indexes/{index_name}/docs/search?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    all_results = []

    with open("api_requests.log", "a", encoding="utf-8") as log_file:
        for filename, embedding in invoice_data:
            body = {
                "vectorQueries": [
                    {
                        "kind": "vector",
                        "vector": embedding,
                        "fields": vector_field,
                        "k": 3
                    }
                ],
                "count": True,
                "select": "title,category",
                "queryType": "semantic",
                "semanticConfiguration": semantic_config,
                "captions": "extractive",
                "answers": "extractive|count-3",
                "queryLanguage": "en-us"
            }

            print(f"🔖 Request for {filename} written to api_requests.log")

            try:
                response = requests.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                results = response.json()

                print(f"\n📄 Results for: {filename}")
                for i, doc in enumerate(results.get("value", [])[:3], start=1):
                    match_title = doc.get("title", "N/A")
                    category = doc.get("category")
                    try:
                        parts = category.split('/')
                        category = parts[4] if len(parts) > 4 else "Unknown"
                    except Exception:
                        category = "Unknown"
                    score = doc.get("@search.score", "N/A")

                    print(f"{i}. 📘 Title: {match_title}, 🏷️ Category: {category}, ⭐ Score: {score}")

                    all_results.append({
                        "Filename": filename,
                        "Rank": i,
                        "PredictedLabel": category,
                        "MatchTitle": match_title,
                        "SimilarityScore": score
                    })

                # ✅ Mark file as searched in DB
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE embeddings SET is_searched = 1 WHERE filename = ? AND is_training = 0", (filename,))
                conn.commit()
                conn.close()

            except Exception as e:
                print(f"❌ Failed to search for {filename}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print("🧾 Response Error:", e.response.text)

    # Save top-3 detailed results
    df = pd.DataFrame(all_results)
    if df.empty:
        print("⚠️ No new files were classified. All files may have already been searched.")
        return
    df.to_csv(RESULT_CSV, index=False)
    print(f"✅ Top 3 results saved to {RESULT_CSV}")

    # Aggregate top-3 for enriched CSV
    summary = (
        df.groupby("Filename")
        .apply(lambda x: pd.Series({
            "TopNeighbors": ", ".join(x["MatchTitle"]),
            "TopSimilarities": ", ".join(f"{float(s):.3f}" for s in x["SimilarityScore"]),
            "PredictedLabel": x["PredictedLabel"].iloc[0]
        }))
        .reset_index()
    )

    summary["Note"] = "Classification via Azure AI Search with semantic vector similarity (Top-3 matches included)"
    summary.to_csv(ENRICHED_CSV, index=False)
    print(f"✅ Enriched prediction file saved to {ENRICHED_CSV}")

# === Optional: Reset Flag ===
def reset_search_flags():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE embeddings SET is_searched = 0 WHERE is_training = 0")
    conn.commit()
    conn.close()
    print("🔁 All test entries reset for reprocessing.")

if __name__ == "__main__":
    classify_invoices_with_vector_search()
    # reset_search_flags()  # Uncomment if needed
