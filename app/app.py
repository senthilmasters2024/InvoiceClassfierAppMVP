import io
import os
import sys
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components
import zipfile

# 🔻 Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.file_handler as file_handler
import classifier as classifier
import visualize_classification
from azuresearchclienttrail import classify_invoices_with_vector_search

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_CSV = BASE_DIR / "results" / "predictions.csv"
AZURE_CSV = BASE_DIR / "results" / "predictions_enriched_with_top3_similarity.csv"
INVOICE_DIR = BASE_DIR / "uploads" / "test"
PLOT_2D_HTML = BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"
PLOT_3D_HTML = BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"
CLASSIFIED_DIR = BASE_DIR / "classified"

st.set_page_config(page_title="Unified Invoice App", layout="wide")

tab1, tab2 = st.tabs(["\U0001F4E5 Upload & Classify", "\U0001F4CA Dashboard"])

# === Function to Create ZIP of All Classified PDFs ===
def create_classified_zip():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for label_folder in CLASSIFIED_DIR.glob("*"):
            if label_folder.is_dir():
                for pdf in label_folder.glob("*.pdf"):
                    arcname = f"{label_folder.name}/{pdf.name}"
                    zipf.write(pdf, arcname)
    zip_buffer.seek(0)
    return zip_buffer

# ================== Tab 1: Upload & Classify ==================
with tab1:
    st.header("\U0001F4DA Upload Training Data")
    label = st.selectbox("Select Label/Category", ["healthcare", "capitalincome", "craftsman", "energy", "other"])
    train_files = st.file_uploader("Upload .pdf or .txt files for training", type=["pdf", "txt"], accept_multiple_files=True, key="train_upload")

    if train_files:
        file_handler.save_training_files(label, train_files)
        st.success(f"✅ Uploaded {len(train_files)} training files to '{label}' category.")

    st.header("\U0001F4C4 Upload Invoices to Classify")
    test_files = st.file_uploader("Upload invoice documents (.pdf only)", type=["pdf"], accept_multiple_files=True, key="test_upload")

    if test_files:
        file_handler.save_test_files(test_files)
        st.success(f"✅ Uploaded {len(test_files)} invoices for classification.")

    st.header("\U0001F4C2 Folder Preview")
    if st.button("Show Uploaded Data"):
        file_handler.show_uploaded_files()

    if st.button("\U0001F50D Train & Classify using KNN"):
        with st.spinner("Generating embeddings and training model..."):
            model = classifier.train_knn_model()
            results, predicted_labels = classifier.classify_test_documents(model)
            classifier.perform_pca()
        st.success("✅ Classification complete!")
        st.write("**Results:**")
        for fname, label in results:
            st.markdown(f"- `{fname}` → **{label}**")

    if st.button("\U0001F4CA Generate Visualizations"):
        with st.spinner("Generating visualizations..."):
            visualize_classification.generate_visualization()
        st.success("✅ Visualizations created. Check the Dashboard tab.")

    if st.button("\U0001F310 Run Azure Vector Search Classification"):
        with st.spinner("Classifying with Azure AI Search..."):
            classify_invoices_with_vector_search()
        st.success("✅ Azure classification complete! Check the Dashboard tab.")

# ================== Tab 2: Dashboard ==================
with tab2:
    st.title("\U0001F4CA Invoice Classification Dashboard")

    @st.cache_data
    def load_data():
        df = pd.read_csv(RESULT_CSV)
        df['Filename'] = df['Filename'].str.replace('.pdf', '', regex=False)
        df['PredictedLabel'] = df['PredictedLabel'].astype(str).str.strip().str.lower()
        return df

    df = load_data()

    # Sidebar selector
    st.sidebar.header("\U0001F4C4 Invoice List")
    search_term = st.sidebar.text_input("\U0001F50D Search Invoice Filename")
    filtered_df = df[df['Filename'].str.contains(search_term, case=False, na=False)] if search_term else df
    selected_invoice = st.sidebar.selectbox("Select an Invoice", filtered_df['Filename'].tolist()) if not filtered_df.empty else None

    if selected_invoice:
        invoice_row = df[df['Filename'] == selected_invoice].iloc[0]
        st.sidebar.markdown(f"**Predicted Label (KNN):** `{invoice_row['PredictedLabel']}`")
        st.sidebar.markdown(f"**Top Neighbors:** `{invoice_row['TopNeighbors']}`")
        if 'TopSimilarities' in invoice_row:
            st.sidebar.markdown(f"**Similarities:** `{invoice_row['TopSimilarities']}`")

    # Visualization
    subtab1, subtab2 = st.tabs(["2D Plot", "3D Plot"])
    with subtab1:
        st.subheader("\U0001F4CD 2D PCA Visualization")
        if PLOT_2D_HTML.exists():
            with open(PLOT_2D_HTML, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
        else:
            st.warning("2D plot not found.")

    with subtab2:
        st.subheader("\U0001F4CD 3D PCA Visualization")
        if PLOT_3D_HTML.exists():
            with open(PLOT_3D_HTML, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=700, scrolling=True, width=1100)
        else:
            st.warning("3D plot not found.")

    # Summary
    st.subheader("\U0001F4CA Classification Summary (KNN)")
    label_counts = df['PredictedLabel'].value_counts()
    fig_summary = px.bar(label_counts, x=label_counts.index, y=label_counts.values,
                         labels={'x': 'Label', 'y': 'Document Count'}, title="Documents per Category")
    st.plotly_chart(fig_summary, use_container_width=True)

    # Azure Comparison Section
    st.subheader("\U0001F9FE Model Comparison: KNN vs Azure AI Search")
    if AZURE_CSV.exists():
        df_azure = pd.read_csv(AZURE_CSV)
        df_merged = pd.merge(df, df_azure, on="Filename", suffixes=("_KNN", "_Azure"))

        st.write("🔁 Merged Predictions")
        st.dataframe(df_merged[["Filename", "PredictedLabel_KNN", "PredictedLabel_Azure", "TopNeighbors_Azure", "TopSimilarities_Azure"]])

        label_diff = (
            df_merged["PredictedLabel_KNN"].astype(str).str.lower() !=
            df_merged["PredictedLabel_Azure"].astype(str).str.lower()
        ).sum()

        st.info(f"🔎 Mismatches between KNN and Azure predictions: **{label_diff}** out of {len(df_merged)}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**KNN Prediction Distribution**")
            fig_knn = px.histogram(df_merged, x="PredictedLabel_KNN", title="KNN")
            st.plotly_chart(fig_knn, use_container_width=True)

        with col2:
            st.markdown("**Azure AI Search Prediction Distribution**")
            fig_azure = px.histogram(df_merged, x="PredictedLabel_Azure", title="Azure AI Search")
            st.plotly_chart(fig_azure, use_container_width=True)

        st.download_button("\U0001F4C5 Download Comparison CSV", df_merged.to_csv(index=False), file_name="KNN_vs_Azure.csv", mime="text/csv")
    else:
        st.warning("Azure AI Search result file not found. Please run Azure classification from Tab 1.")

    st.subheader("\U0001F4E5 Export Predictions as CSV")
    st.download_button("\U0001F4C4 Download Full Predictions CSV", df.to_csv(index=False), file_name="classified_invoices.csv", mime="text/csv")
