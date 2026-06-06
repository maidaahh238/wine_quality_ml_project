#  Wine Quality Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Phase](https://img.shields.io/badge/Phase-4%20of%204-purple)

> Predicting wine quality using chemical properties through a complete 4-phase ML pipeline — from raw data to model evaluation.

---

##  Project Overview

This project analyzes a dataset of **600 wine samples**, each described by 9 chemical properties. The goal is to predict wine quality using multiple machine learning algorithms and compare their performance.

| Property | Value |
|---|---|
| Dataset | Wine Quality (600 rows, 9 features) |
| Target | Quality rating (numeric → binary classification) |
| Algorithms Used | Linear Regression, Logistic Regression, KNN, K-Means |
| Tools | Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn |

---

##  Project Structure

```
wine-quality-ml/
├── 23set029_Project_Phase04_ML.ipynb   ← Main notebook (all 4 phases)
├── wine_dataset_raw_600_rows.csv       ← Raw dataset (upload when running)
├── README.md                           ← You are here
```

---

##  Project Phases

### Phase 1 — Data Loading & Exploration
- Loaded raw CSV with 600 rows and 9 chemical feature columns
- Explored data types, null counts, and basic statistics with `df.info()` and `df.describe()`

### Phase 2 — Data Cleaning & Preprocessing
- Cleaned column names (lowercase, underscores)
- Stripped non-numeric characters from all columns
- Handled missing values using **median imputation**
- Removed duplicate rows
- Removed outliers using the **IQR method**
- Saved cleaned dataset as `wine_cleaned.csv`

### Phase 3 — Visualization
- Scatter plots (Fixed Acidity vs Quality)
- Histograms for all features
- **Correlation heatmap** using Seaborn
- Regression line plots
- Actual vs Predicted comparison charts
- **Gradient Descent** loss curve (converged from 32.25 → 2.74)
- Confusion matrices for classification models

### Phase 4 — Model Training & Evaluation

####  Linear Regression
| Metric | Score |
|---|---|
| MAE | 1.43 |
| MSE | 2.66 |
| RMSE | 1.63 |
| R² | 0.021 |

####  Logistic Regression (Binary: quality ≥ 6)
| Metric | Score |
|---|---|
| Accuracy | 50.8% |
| Precision | 46.4% |
| Recall | 22.8% |

####  K-Nearest Neighbors (k=5)
| Metric | Score |
|---|---|
| Accuracy | 48.3% |
| Precision | 45.6% |
| Recall | 45.6% |
| F1 Score | 45.6% |

####  K-Means Clustering (Unsupervised, k=3)
| Metric | Score |
|---|---|
| WCSS (Inertia) | 4040.6 |
| MAE | 4.47 |
| MSE | 23.22 |

---

##  Key Findings

- **Total sulfur dioxide** had the highest correlation with quality (0.08) — overall features were weakly correlated with quality, making this a challenging prediction task
- **Gradient Descent** successfully converged, reducing loss from **32.26 → 2.74** over 200 epochs
- Classification models struggled (~50% accuracy), suggesting wine quality is influenced by subjective factors beyond chemical properties alone
- **K-Means** clustering with Elbow Method identified 3 natural wine clusters

---

##  How to Run

1. Open the notebook in **Google Colab** (recommended) or Jupyter Notebook
2. Upload `wine_dataset_raw_600_rows.csv` when prompted
3. Run all cells top to bottom (`Runtime → Run all`)

```bash
# Or locally
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook 23set029_Project_Phase04_ML.ipynb
```

---

##  Skills Demonstrated

- `Data Cleaning` — handling nulls, duplicates, outliers, type conversion
- `EDA` — correlation analysis, distribution plots, heatmaps
- `Supervised Learning` — Linear Regression, Logistic Regression, KNN
- `Unsupervised Learning` — K-Means Clustering, Elbow Method
- `Model Evaluation` — MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1, Confusion Matrix
- `Gradient Descent` — implemented from scratch with loss tracking

---

##  Author

Maidah Javed
Roll No: 23SET029
BS Software Engineering | [Punjab Tianjin University of Technology]


---

##  License

This project is for academic and portfolio purposes.
