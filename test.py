import ast
import sqlite3
import requests
import json
from pathlib import Path

# === Setup ===
api_key = "66ni9vhfflFq2xifbrTS5vUAGUE4sZyMlpXKff6TEiAzSeCXePXm"
search_service = "classificationaisearch1"
index_name = "classificationrag1-1748553314805"
vector_field = "text_vector"
vector_profile = "classificationrag1-1748553314805-azureOpenAi-text-profile"  # e.g., "classificationrag1-1748553314805-azureOpenAi-text-profile"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "results" / "embeddings_store.db"

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
                data.append((filename, embedding))
        except Exception as e:
            print(f"Could not parse embedding for {filename}: {e}")
    return data

def classify_invoices_with_rest():
    invoice_data = get_new_invoice_embeddings()
    endpoint = f"https://{search_service}.search.windows.net/indexes/{index_name}/docs/search?api-version=2024-05-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    for filename, embedding in invoice_data:
        print(f"\n🔎 Searching for: {filename}")
        body = {
            "vectorQueries": [
                {
                    "kind": "vector",
                    "value": embedding,
                    "fields": vector_field,
                    "k": 3,
                    "profile": vector_profile
                }
            ]
        }
        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(body))
            response.raise_for_status()
            results = response.json()
            # Print matches and scores
            for doc in results.get("value", []):
                print("📄 Match:", doc.get("chunk", "No content"))
                print("⭐ Score:", doc.get("@search.score", "No score"))
        except Exception as e:
            print(f"❌ Failed to search for {filename}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(e.response.text)

if __name__ == "__main__":
    classify_invoices_with_rest()
