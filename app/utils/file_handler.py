import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
TRAIN_DIR = UPLOAD_DIR / "train"
TEST_DIR = UPLOAD_DIR / "test"

def save_training_files(label, files):
    label_path = TRAIN_DIR / label
    label_path.mkdir(parents=True, exist_ok=True)
    for file in files:
        with open(label_path / file.name, "wb") as f:
            f.write(file.getbuffer())

def save_test_files(files):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        with open(TEST_DIR / file.name, "wb") as f:
            f.write(file.getbuffer())

def show_uploaded_files():
    st.subheader("Training Files")
    for folder in TRAIN_DIR.iterdir():
        files = list(folder.iterdir())
        if files:
            st.markdown(f"**{folder.name}** ({len(files)} files)")
            for f in files:
                st.markdown(f"- {f.name}")

    st.subheader("Invoices to Classify")
    for f in TEST_DIR.iterdir():
        st.markdown(f"- {f.name}")
