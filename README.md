# Dynamic-credit-scoring
Dynamic credit scoring engine using real-world financial data and interpretable ML models, enabling real-time risk reclassification and actionable customer insights through SHAP and behavioral segmentation.
---

## 📦 Features
- Credit default prediction (binary classification)
- SHAP explainability per transaction
- Customer segmentation using KMeans
- Real-time scoring simulation
- Streamlit dashboard to visualize risk

---

## 📁 Folder Structure
```
dynamic_credit_scoring/
├── data/                  # Raw and processed data
├── src/                   # Core logic for training, scoring, explaining
├── app/                   # Streamlit dashboard
├── notebooks/             # EDA and modeling notebooks
├── requirements.txt       # Project dependencies
├── config.yaml            # Configurable paths and model params
└── README.md
```

---

## 🔧 Setup
```bash
git clone https://github.com/yourusername/dynamic_credit_scoring.git
cd dynamic_credit_scoring
pip install -r requirements.txt
```

---

## ⚙️ Usage
```bash
# Step 1: Clean the data
python src/preprocess.py

# Step 2: Train model
python src/train_model.py

# Step 3: Score new customers
python src/score.py

# Step 4: Generate SHAP values
python src/explain.py

# Step 5: Segment customers
python src/segment.py

# Step 6: Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 📊 Streamlit Dashboard
- Upload new customers as CSV
- View credit risk scores
- Explore SHAP waterfall plots
- Visualize and understand feature impact

---

## 🧪 Notebooks
- `01_EDA.ipynb`: Data profiling and distribution
- `02_Modeling.ipynb`: Training and evaluation
- `03_SHAP_Explainability.ipynb`: Visual interpretation

---

## ✅ Resume-Ready One-Liner
> Developed a dynamic credit scoring engine using real-world financial data and interpretable ML models, enabling real-time risk reclassification and actionable customer insights through SHAP and behavioral segmentation.

---

## 📚 Dataset
[Give Me Some Credit – Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data)

Download `cs-training.csv` and place it in `data/`.

---
