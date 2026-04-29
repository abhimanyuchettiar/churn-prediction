# Customer Churn Prediction & Segmentation Dashboard

End-to-end machine learning pipeline to predict customer churn, explain predictions using SHAP, and segment high-risk customers using K-Means clustering. Includes an interactive Plotly Dash dashboard deployed on Streamlit Cloud.

## Tech Stack
Python, Scikit-learn, XGBoost, SHAP, Pandas, NumPy, Plotly Dash, Streamlit

## Project Structure
```
churn-prediction/
├── data/
│   ├── raw/              # Original Telco Churn dataset (from Kaggle)
│   └── processed/        # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Cleaning and feature engineering
│   ├── 03_modeling.ipynb         # Model training and comparison
│   ├── 04_shap_explainability.ipynb
│   └── 05_segmentation.ipynb     # K-Means clustering on high-risk customers
├── src/
│   ├── data/
│   │   ├── load_data.py          # Dataset loading utilities
│   │   └── preprocess.py         # Cleaning and encoding pipeline
│   ├── models/
│   │   ├── train.py              # Model training (LR, RF, XGBoost)
│   │   ├── evaluate.py           # ROC-AUC, confusion matrix, classification report
│   │   └── shap_explainer.py     # SHAP value computation
│   └── visualization/
│       └── plots.py              # Reusable Plotly chart functions
├── dashboard/
│   └── app.py                    # Plotly Dash / Streamlit dashboard
├── models/
│   └── .gitkeep                  # Saved model artifacts (.pkl files)
├── reports/
│   └── figures/                  # Exported charts and SHAP plots
├── requirements.txt
├── .gitignore
└── README.md
```

## Results
| Model | ROC-AUC | Precision | Recall |
|---|---|---|---|
| Logistic Regression | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |

## Setup
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
```

## Dashboard
Live demo: [link here]
