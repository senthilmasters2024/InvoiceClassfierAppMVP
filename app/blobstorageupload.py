import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceExistsError

# === Step 1: Setup Config ===
CONTAINER_NAME = "training-data"
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_TRAINING_DIR = BASE_DIR/"./uploads/train"

# === Step 2: Create BlobServiceClient and ContainerClient ===
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# === Step 3: Create container safely ===
try:
    container_client.create_container()
    print(f"✅ Created container: {CONTAINER_NAME}")
except ResourceExistsError:
    print(f"ℹ️ Container '{CONTAINER_NAME}' already exists. Continuing...")

# === Step 4: Upload training files ===
for root, _, files in os.walk(LOCAL_TRAINING_DIR):
    for file_name in files:
        local_path = os.path.join(root, file_name)
        blob_path = os.path.relpath(local_path, LOCAL_TRAINING_DIR).replace("\\", "/")  # keep folder structure

        blob_client = container_client.get_blob_client(blob_path)

        with open(local_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type="application/pdf")
            )
            print(f"📤 Uploaded: {blob_path}")

print("🎉 All files uploaded successfully.")
