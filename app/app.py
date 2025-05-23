import os
import sys
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components

# 👇 Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.file_handler as file_handler
import classifier as classifier
import visualize_classification

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_CSV = BASE_DIR / "results" / "predictions.csv"
INVOICE_DIR = BASE_DIR / "uploads" / "test"
PLOT_2D_HTML = BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"
PLOT_3D_HTML = BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"

st.set_page_config(page_title="Unified Invoice App", layout="wide")

tab1, tab2 = st.tabs(["📥 Upload & Classify", "📊 Dashboard"])

# ================== Tab 1: Upload & Classify ==================
with tab1:
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

    if st.button("📊 Generate Visualizations"):
        with st.spinner("Generating visualizations..."):
            visualize_classification.generate_visualization()
        st.success("✅ Visualizations created. Check the Dashboard tab.")

# ================== Tab 2: Dashboard ==================
with tab2:
    st.title("📊 Invoice Classification Dashboard")

    @st.cache_data
    def load_data():
        df = pd.read_csv(RESULT_CSV)
        df['Filename'] = df['Filename'].str.replace('.pdf', '', regex=False)
        df['PredictedLabel'] = df['PredictedLabel'].astype(str).str.strip().str.lower()
        return df

    df = load_data()

    # Sidebar selector
    st.sidebar.header("📄 Invoice List")
    search_term = st.sidebar.text_input("🔍 Search Invoice Filename")
    filtered_df = df[df['Filename'].str.contains(search_term, case=False, na=False)] if search_term else df
    if filtered_df.empty:
        st.sidebar.warning("No matching invoices found.")
        selected_invoice = None
    else:
        selected_invoice = st.sidebar.selectbox("Select an Invoice", filtered_df['Filename'].tolist())

    if selected_invoice:
        invoice_row = df[df['Filename'] == selected_invoice].iloc[0]
        st.sidebar.markdown(f"**Predicted Label:** `{invoice_row['PredictedLabel']}`")
        st.sidebar.markdown(f"**Top Neighbors:** `{invoice_row['TopNeighbors']}`")
        if 'TopSimilarities' in invoice_row:
            st.sidebar.markdown(f"**Similarities:** `{invoice_row['TopSimilarities']}`")

        invoice_path = INVOICE_DIR / f"{selected_invoice}.pdf"
        if invoice_path.exists():
            st.subheader("📄 Selected Invoice Preview")
            base64_pdf = base64.b64encode(invoice_path.read_bytes()).decode('utf-8')
            st.components.v1.html(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>', height=520)

        # Neighbor
        if 'TopNeighbors' in invoice_row and isinstance(invoice_row['TopNeighbors'], str):
            top_neighbor = invoice_row['TopNeighbors'].split(';')[0].strip()
            neighbor_path = INVOICE_DIR / top_neighbor
            if neighbor_path.exists():
                st.subheader("📎 Top Neighbor Invoice Preview")
                base64_neighbor = base64.b64encode(neighbor_path.read_bytes()).decode('utf-8')
                st.components.v1.html(f'<iframe src="data:application/pdf;base64,{base64_neighbor}" width="100%" height="500"></iframe>', height=520)

    # Visualization
    subtab1, subtab2 = st.tabs(["2D Plot", "3D Plot"])
    with subtab1:
        st.subheader("📍 2D PCA Visualization")
        if PLOT_2D_HTML.exists():
            with open(PLOT_2D_HTML, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
        else:
            st.warning("2D plot not found.")

    with subtab2:
        st.subheader("📍 3D PCA Visualization")
        if PLOT_3D_HTML.exists():
            with open(PLOT_3D_HTML, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=700, scrolling=True, width=1100)
        else:
            st.warning("3D plot not found.")

    # Summary
    st.subheader("📊 Classification Summary")
    label_counts = df['PredictedLabel'].value_counts()
    fig_summary = px.bar(label_counts, x=label_counts.index, y=label_counts.values,
                         labels={'x': 'Label', 'y': 'Document Count'}, title="Documents per Category")
    st.plotly_chart(fig_summary, use_container_width=True)

    st.subheader("📥 Export Results")
    st.download_button("Download Full Predictions CSV", df.to_csv(index=False), file_name="classified_invoices.csv", mime="text/csv")
