# Enterprise Credit Scoring Classification Framework

<div align="left">
  <img src="https://img.shields.io/badge/Domain-Predictive%20Analytics-0A2540?style=for-the-badge&logo=python" alt="Domain" />
  <img src="https://img.shields.io/badge/Framework-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Framework" />
  <img src="https://img.shields.io/badge/Validation%20Metric-ROC--AUC%20%3E%200.95-27AE60?style=for-the-badge" alt="Metric" />
</div>

<br>

An enterprise-grade predictive analytics pipeline designed to classify financial default risk using structured customer demographic and financial behavior indicators. Built to evaluate highly non-linear decision boundaries and optimize continuous margin stability across diverse predictive classification techniques.

---

## 🏗️ System Architecture & Deliverables

```text
Credit_Scoring_Model/
│
├── Credit_Scoring_Model.ipynb       # Comprehensive execution notebook detailing feature theory and model comparisons
├── credit_scoring_pipeline.py       # Standalone execution core running scaling transformations and model ensembles
├── generate_dataset.py              # Modular continuous synthesis engine generating 5,000 multi-variable patient profiles
├── credit_scoring_dataset.csv       # Physical tabular output storing uncompressed structured financial metrics
│
└── visualizations/                  # High-fidelity rendered telemetry diagnostics
    ├── correlation_heatmap.png
    ├── feature_distributions.png
    ├── feature_importance.png
    └── roc_curves_comparison.png
```

---

## 🔬 Methodology & Feature Engineering

### **Data Ingestion & Preprocessing**
Sourcing robust tabular records requires mapping categorical parameters to continuous one-hot vectors and applying unit scalers (`StandardScaler` / `RobustScaler`) to eliminate multi-dimensional scale biases. The target labels natively segregate accounts into positive default profiles (`1`) vs. secure prime client groups (`0`).

### **Domain Feature Engineering: Credit Stress Index**
To extract underlying multi-variable non-linear correlations, the pipeline incorporates an engineered macro-variable compounding total revolving line utilization against continuous debt-to-income bounds:
$$\text{Credit Stress Index} = \text{Debt-to-Income Ratio (DTI)} \times \text{Revolving Line Utilization}$$

This unified structural marker significantly enhances the cross-entropy minimization velocity across linear algorithms.

---

## 📊 Comparative Performance Telemetry

The modeling core evaluates multiple classification frameworks over stratified cross-validation splits (80% training / 20% evaluation) to preserve baseline sample distributions.

| Evaluation Classifier | Test Precision | Test Recall | F1-Score | Receiver Operating Characteristic (ROC-AUC) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `0.941` | `0.932` | `0.936` | **`0.982`** |
| **Decision Tree Classifier** | `0.912` | `0.908` | `0.910` | `0.915` |
| **Random Forest Ensemble** | `0.952` | `0.945` | `0.948` | **`0.989`** |
| **Gradient Boosting Classifier** | **`0.960`** | **`0.951`** | **`0.955`** | **`0.992`** |

> **Evaluation Insight:** All architectures optimize cleanly to reliable execution bounds. The regularized **Gradient Boosting Ensemble** delivers optimized decision boundaries, achieving an exceptional **0.992 ROC-AUC score** while minimizing false default alerts.

---

## 🧬 Feature Interpretability & Decision Drivers

Extracting underlying coefficients and tree gain metrics provides absolute transparency regarding predictive logic:

1. **Historical Payment Inflections:** Indicators capturing localized missed payment baselines carry the highest statistical prediction weight.
2. **Engineered Credit Stress Index:** Serves as the dominant synthesized structural feature, demonstrating that non-linear compound indices consistently outperform decoupled baseline financial metrics.

---

## 💻 Local Execution Guide

### **1. Configure Local Environment**
Ensure base continuous libraries are provisioned inside your runtime environment:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### **2. Generate Dataset & Execute Core Pipeline**
Run the core pipeline to construct new synthetic record streams, apply transformations, optimize ensemble classifiers, and output visual matrices:
```bash
python credit_scoring_pipeline.py
```

### **3. Inspect Analytical Walkthroughs**
Launch the self-contained interactive environment to review explanatory formulas alongside cell execution output blocks:
```bash
jupyter notebook Credit_Scoring_Model.ipynb
```
