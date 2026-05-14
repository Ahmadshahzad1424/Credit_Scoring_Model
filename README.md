# 📊 Machine Learning Task 1: Credit Scoring Model

<div align="center">
  <img src="https://img.shields.io/badge/Domain-Machine%20Learning-blue?style=for-the-badge&logo=python" alt="Domain" />
  <img src="https://img.shields.io/badge/Task-01-success?style=for-the-badge" alt="Task" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" alt="Status" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
</div>

<br>

A state-of-the-art implementation of an individual **Credit Scoring Model** to predict customer creditworthiness using classification algorithms on financial history data. Designed as **Task 1** for the **Machine Learning Internship**.

---

## 🚀 Repository Structure

```text
Credit_Scoring_Model/
│
├── Credit_Scoring_Model.ipynb     # Comprehensive Jupyter Notebook with end-to-end markdown explanations
├── credit_scoring_pipeline.py     # Standalone Python modular pipeline script (Execution ready)
├── generate_dataset.py            # Companion script generating realistic synthetic customer financial data
├── credit_scoring_dataset.csv     # Self-contained dataset of 5,000 customers with class imbalances & noise
│
└── visualizations/                # High-fidelity rendered outputs
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    ├── feature_importance.png
    └── roc_curves_comparison.png
```

---

## 🎯 Task Objective & Approach

### **Objective**
Predict whether an individual is **Creditworthy (Approved)** or **High Risk (Default)** using historical demographic and liability markers.

### **Approach**
1. **Data Preprocessing Pipeline:** Standard Scaling for continuous numerical fields and One-Hot Encoding for categorical features (`loan_purpose`).
2. **Advanced Feature Engineering:** Formulated a non-linear interaction compound metric named the **Credit Stress Index** combining overall debt-to-income and revolving line utilization.
3. **Model Comparison Suite:** Evaluated linear classification baselines against highly robust decision ensembles across a rigorous validation splits framework.

---

## 🧪 Algorithms Compared & Evaluation Metrics

Each model was trained on an 80/20 stratified split preserving real-world target class proportions. 

| Model | Precision | Recall | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `0.916` | `0.957` | `0.936` | **`0.959`** |
| **Decision Tree** | `0.898` | `0.932` | `0.915` | `0.922` |
| **Random Forest** | `0.885` | `0.954` | `0.918` | `0.951` |
| **Gradient Boosting** | `0.903` | `0.953` | `0.927` | **`0.955`** |

> **Insight:** Both **Logistic Regression** (highly suited for scaled credit scoring cards) and **Gradient Boosting Ensembles** delivered stellar performances exceeding **0.95 ROC-AUC**, successfully segregating default patterns.

---

## 🔬 Model Interpretability & Feature Drivers

Extracting feature importances from the best ensemble models provided deep transparency into credit approval drivers:

1. **Missed Payments History (`0.455` weight):** The overwhelming historical predictor of financial defaults.
2. **Credit Stress Index (`0.207` weight):** Our custom engineered domain metric provided an immense lift, proving twice as powerful as raw length of employment or baseline income.
3. **Employment Length & Credit Utilization:** Provide supplementary but critical contextual bounds to customer risk evaluation.

---

## 💻 How to Run Locally

### **1. Install Dependencies**
Ensure you have a Python environment equipped with core scientific libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **2. Execute Modular Pipeline Directly**
Run the core pipeline to ingest data, fit models, output evaluation results, and automatically re-render all visual graphs:
```bash
python credit_scoring_pipeline.py
```

### **3. Launch Interactive Notebook**
Open the pre-built notebook to view cell-by-cell walkthroughs, markdown instructions, and embedded graphics:
```bash
jupyter notebook Credit_Scoring_Model.ipynb
```

---

## 📌 Submission Instructions Compliance
- **Source Code Ready:** Formatted completely for direct upload to GitHub inside repository `Credit_Scoring_Model`.
- **Reproducibility Guaranteed:** Companion generator script handles offline or fresh generation flawlessly.
- **Metrics Fulfilled:** Fully demonstrates Precision, Recall, F1-Score, and ROC-AUC visualizations.

<br>
<div align="center">
  <b>Built with ❤️ for Machine Learning Internship Program</b>
</div>
