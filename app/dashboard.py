import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_CSV = BASE_DIR / "results" / "predictions.csv"
INVOICE_DIR = BASE_DIR / "uploads" / "test"
PLOT_2D_HTML = BASE_DIR / "results" / "2D_Predicted_vs_Train_Similarity.html"
PLOT_3D_HTML = BASE_DIR / "results" / "3D_Predicted_vs_Train_Similarity.html"

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(RESULT_CSV)
    df['Filename'] = df['Filename'].str.replace('.pdf', '', regex=False)
    df['PredictedLabel'] = df['PredictedLabel'].astype(str).str.strip().str.lower()
    return df

df = load_data()

st.title("📊 Invoice Classification Dashboard")

# Sidebar - Invoice Selector with Search
st.sidebar.header("📄 Invoice List")
search_term = st.sidebar.text_input("🔍 Search Invoice Filename")
filtered_df = df[df['Filename'].str.contains(search_term, case=False, na=False)] if search_term else df
if filtered_df.empty:
    st.sidebar.warning("No matching invoices found.")
    selected_invoice = None
else:
    selected_invoice = st.sidebar.selectbox("Select an Invoice", filtered_df['Filename'].tolist())

# Display invoice details
invoice_row = df[df['Filename'] == selected_invoice].iloc[0]
st.sidebar.markdown(f"**Predicted Label:** `{invoice_row['PredictedLabel']}`")
st.sidebar.markdown(f"**Top Neighbors:** `{invoice_row['TopNeighbors']}`")
if 'TopSimilarities' in invoice_row:
    st.sidebar.markdown(f"**Similarities:** `{invoice_row['TopSimilarities']}`")

invoice_path = INVOICE_DIR / f"{selected_invoice}.pdf"
neighbor_path = None
if 'TopNeighbors' in invoice_row and isinstance(invoice_row['TopNeighbors'], str):
    top_neighbor = invoice_row['TopNeighbors'].split(';')[0].strip()
    neighbor_path = INVOICE_DIR / top_neighbor if top_neighbor else None

if invoice_path.exists():
    st.subheader("📄 Selected Invoice Preview")
    st.markdown(f'<embed src="{invoice_path.as_uri()}" width="100%" height="600px" type="application/pdf">',
                unsafe_allow_html=True)
    with open(invoice_path, "rb") as pdf_file:
        st.sidebar.download_button(
            label="📥 Download PDF",
            data=pdf_file,
            file_name=invoice_path.name,
            mime="application/pdf"
        )

if neighbor_path and neighbor_path.exists():
    st.subheader("📎 Top Neighbor Invoice Preview")
    base64_neighbor = base64.b64encode(neighbor_path.read_bytes()).decode('utf-8')
    st.components.v1.html(f'<iframe src="data:application/pdf;base64,{base64_neighbor}" width="100%" height="500"></iframe>', height=520)
    with open(neighbor_path, "rb") as pdf_file:
        st.sidebar.download_button(
            label="📥 Download Top Neighbor PDF",
            data=pdf_file,
            file_name=neighbor_path.name,
            mime="application/pdf"
        )

# Tabs for visualization
tab1, tab2 = st.tabs(["2D Plot", "3D Plot"])
with tab1:
    st.subheader("📍 2D PCA Visualization")
    if PLOT_2D_HTML.exists():
        with open(PLOT_2D_HTML, 'r', encoding='utf-8') as f:
            html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
    else:
        st.warning("2D plot not found. Please generate it first.")

with tab2:
    st.subheader("📍 3D PCA Visualization")
    if PLOT_3D_HTML.exists():
        with open(PLOT_3D_HTML, 'r', encoding='utf-8') as f:
            html_content = f.read()
        html_content = html_content.replace("Plotly.newPlot(", "Plotly.newPlot(", 1).replace(
            ", {", ", {margin: {r: 140}, ", 1)
        components.html(html_content, height=700, scrolling=True, width=1100)
    else:
        st.warning("3D plot not found. Please generate it first.")

# Classification summary
st.subheader("📊 Classification Summary")
label_counts = df['PredictedLabel'].value_counts()
fig_summary = px.bar(label_counts, x=label_counts.index, y=label_counts.values,
                     labels={'x': 'Label', 'y': 'Document Count'}, title="Documents per Category")
st.plotly_chart(fig_summary, use_container_width=True)

# Download filtered result
st.subheader("📥 Export Results")
st.download_button("Download Full Predictions CSV", df.to_csv(index=False), file_name="classified_invoices.csv", mime="text/csv")
