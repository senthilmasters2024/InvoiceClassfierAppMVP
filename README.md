# 🧾 Invoice Classification Dashboard

A Streamlit-powered application to upload, classify, and visualize PDF/text-based invoices using OpenAI embeddings and K-Nearest Neighbors (KNN). The app supports training on custom categories, visualizing results in 2D/3D space, and exporting classification summaries.

---

## 🚀 Features

- ✅ Upload training data by category
- ✅ Upload and classify invoices using OpenAI embeddings + KNN
- ✅ PDF preview and top-neighbor visualization
- ✅ 2D and 3D PCA plots for semantic embedding spaces
- ✅ CSV export of classification results
- ✅ Interactive dashboard (search, filter, view)

---

## 🐳 Docker Setup

### 🔧 Build the Docker image

```bash
docker build -t invoice-classifier .
```

### ▶️ Run the Docker container

```bash
docker run -p 8501:8501 invoice-classifier
```

Then open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 📁 Folder Structure

```
.
├── app/
│   ├── app.py                    # Unified main Streamlit app
│   ├── classifier.py             # Embedding + KNN logic
│   ├── visualize_classification.py
│   └── utils/
│       ├── file_handler.py       # File management logic
│       └── pdf_text.py           # Text extraction from PDFs
├── uploads/
│   ├── train/                    # Uploaded training files
│   └── test/                     # Uploaded invoices to classify
├── results/
│   ├── predictions.csv           # Classification results
│   ├── 2D_Predicted_vs_Train_Similarity.html
│   └── 3D_Predicted_vs_Train_Similarity.html
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

Your `requirements.txt` should include:

```
streamlit
pandas
plotly
scikit-learn
PyMuPDF
```

---

## ☁️ Cloud Deployment (Render.com Recommended)

1. Push the project to GitHub
2. Go to [https://render.com](https://render.com)
3. Click **New → Web Service**
4. Choose **Docker** and your GitHub repo
5. Set port to `8501`
6. Deploy!

---

## 📌 Notes

- This app uses OpenAI embeddings under the hood. You’ll need an API key (stored securely — not included here).
- Make sure uploaded PDFs are text-based (not image-scanned), or integrate OCR.

---

## ✨ Demo Screenshot (Optional)
![Dashboard Preview](./preview.png)

---

## 📃 License

MIT License — free to use and modify.

---

## 👨‍💻 Author

Built by [Your Name]  
[LinkedIn](https://linkedin.com/in/yourprofile) • [GitHub](https://github.com/yourprofile)
