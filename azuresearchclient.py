
import ast
import sqlite3
import sys
import json
import requests
import pandas as pd
from pathlib import Path

print("Python EXE:", sys.executable)

# === Setup ===
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "results" / "embeddings_store.db"
RESULT_CSV = BASE_DIR / "results" / "top3_predictions_with_scores.csv"
RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)

api_key = "66ni9vhfflFq2xifbrTS5vUAGUE4sZyMlpXKff6TEiAzSeCXePXm"
search_service = "classificationaisearch1"
index_name = "classificationrag1-1748553314805"
vector_field = "text_vector"
semantic_config = "classificationrag1-1748553314805-semantic-configuration"
api_version = "2025-05-01-preview"

def get_new_invoice_embeddings(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, embedding FROM embeddings WHERE is_training = 0")
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
                "select": "title, content, category",
                "queryType": "semantic",
                "semanticConfiguration": semantic_config,
                "captions": "extractive",
                "answers": "extractive|count-3",
                "queryLanguage": "en-us"
            }

            request_log = (
                f"\n--- Request for: {filename} ---\n"
                f"Endpoint: {endpoint}\n"
                f"Headers: {json.dumps(headers)}\n"
                f"Body sample: {json.dumps(body)[:400]}...\n"
                f"{'-' * 50}\n"
            )
            log_file.write(request_log)
            print(f"🔖 Request for {filename} written to api_requests.log")

            try:
                response = requests.post(endpoint, headers=headers, json=body)
                response.raise_for_status()
                results = response.json()

                print(f"\n📄 Results for: {filename}")
                for i, doc in enumerate(results.get("value", [])[:3], start=1):
                    match_title = doc.get("title", "N/A")
                    category = doc.get("category", "N/A")
                    score = doc.get("@search.score", "N/A")
                    print(f"{i}. 📘 Title: {match_title}, 🏷️ Category: {category}, ⭐ Score: {score}")

                    all_results.append({
                        "filename": filename,
                        "rank": i,
                        "match_title": match_title,
                        "category": category,
                        "score": score
                    })

            except Exception as e:
                print(f"❌ Failed to search for {filename}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print("🧾 Response Error:", e.response.text)

    # Save results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_CSV, index=False)
    print(f"✅ Top 3 results saved to {RESULT_CSV}")

if __name__ == "__main__":
    classify_invoices_with_vector_search()
