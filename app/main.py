import os
import sys
import streamlit as st
import visualize_classification

# 👇 Add root to system path so 'app' module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Now imports work
from app.utils import file_handler
import app.classifier as classifier
st.set_page_config(page_title="Invoice Classifier Uploader", layout="centered")
st.title("📄 Invoice Classifier Uploader")
st.header("📚 Upload Training Data")
label = st.selectbox("Select Label/Category", ["healthcare", "finance", "water", "energy", "other"])
train_files = st.file_uploader("Upload .pdf or .txt files for training", type=["pdf", "txt"], accept_multiple_files=True, key="train_upload")

if train_files:
    file_handler.save_training_files(label, train_files)
    st.success(f"✅ Uploaded {len(train_files)} training files to '{label}' category.")

st.header("📄 Upload Invoices to Classify")
test_files = st.file_uploader("Upload invoice documents (.pdf only)", type=["pdf"], accept_multiple_files=True, key="test_upload")

if test_files:
    file_handler.save_test_files(test_files)
    st.success(f"✅ Uploaded {len(test_files)} invoices for classification.")

st.header("📂 Folder Preview")
if st.button("Show Uploaded Data"):
    file_handler.show_uploaded_files()

if st.button("🔍 Train & Classify"):
    with st.spinner("Generating embeddings and training model..."):
        model = classifier.train_knn_model()
        results, predicted_labels = classifier.classify_test_documents(model)
    st.success("✅ Classification complete!")
    st.write("**Results:**")
    for fname, label in results:
        st.markdown(f"- `{fname}` → **{label}**")

st.info("After uploading, you can now run your Python classification script.")

# ✅ New Button to trigger visualization
if st.button("📊 Generate & View Visualizations"):
    with st.spinner("Generating visualizations..."):
        visualize_classification.generate_visualization()
    st.success("✅ Visualization files created! Check the `results/` folder for 2D and 3D plots.")
