import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Set aesthetic styling for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def load_and_explore_data(filepath='credit_scoring_dataset.csv', vis_dir='visualizations'):
    """
    Loads the credit scoring dataset, performs exploratory data analysis,
    and saves presentation-quality figures.
    """
    os.makedirs(vis_dir, exist_ok=True)
    print("=" * 70)
    print("STEP 1: Data Loading & Exploratory Data Analysis (EDA)")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"Dataset Loaded Successfully. Shape: {df.shape}\n")
    print("First 5 rows preview:")
    print(df.head().to_string())
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    # Custom diverging palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Feature Correlation Matrix", pad=20)
    plt.tight_layout()
    heatmap_path = os.path.join(vis_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f" Saved correlation heatmap to '{heatmap_path}'")
    
    # 2. Key Features Distribution by Target Class
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data=df, x='debt_to_income_ratio', hue='creditworthiness_target', 
                 kde=True, bins=30, ax=axes[0], palette={1: '#2ecc71', 0: '#e74c3c'}, multiple="stack")
    axes[0].set_title("Debt-to-Income Ratio Distribution")
    
    sns.histplot(data=df, x='credit_utilization_ratio', hue='creditworthiness_target', 
                 kde=True, bins=30, ax=axes[1], palette={1: '#2ecc71', 0: '#e74c3c'}, multiple="stack")
    axes[1].set_title("Credit Utilization Ratio Distribution")
    
    sns.countplot(data=df, x='missed_payments_history', hue='creditworthiness_target', 
                  ax=axes[2], palette={1: '#2ecc71', 0: '#e74c3c'})
    axes[2].set_title("Missed Payments History Impact")
    
    plt.tight_layout()
    dist_path = os.path.join(vis_dir, 'feature_distributions.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f" Saved feature distributions plot to '{dist_path}'\n")
    
    return df

def preprocess_and_engineer_features(df):
    """
    Preprocesses the data:
    - Drops ID column
    - Applies custom domain feature engineering
    - Prepares column transformers for scaling and encoding
    """
    print("=" * 70)
    print("STEP 2: Feature Engineering & Preprocessing Pipeline")
    print("=" * 70)
    
    # Feature Engineering: Create a compound 'Credit Stress Index'
    # Interaction of debt burden and line utilization
    df['credit_stress_index'] = df['debt_to_income_ratio'] * df['credit_utilization_ratio']
    print(" Engineered new domain feature: 'credit_stress_index' (DTI * Credit Utilization)")
    
    # Define features and target
    X = df.drop(columns=['customer_id', 'creditworthiness_target'])
    y = df['creditworthiness_target']
    
    # Identify feature types
    numeric_features = ['age', 'annual_income', 'employment_length_years', 
                        'total_debt_outstanding', 'debt_to_income_ratio', 
                        'credit_utilization_ratio', 'num_open_accounts', 
                        'missed_payments_history', 'loan_amount_requested', 
                        'credit_stress_index']
    
    categorical_features = ['loan_purpose']
    
    # Build robust column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Perform split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f" Training Set: {X_train.shape[0]} samples")
    print(f" Testing Set:  {X_test.shape[0]} samples\n")
    
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, vis_dir='visualizations'):
    """
    Trains multiple baseline and advanced classification algorithms,
    evaluates performance across specified metrics, and extracts feature importances.
    """
    print("=" * 70)
    print("STEP 3: Model Training, Evaluation & Comparison")
    print("=" * 70)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    results = []
    roc_curves = {}
    fitted_pipelines = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training [{name}]...")
        # Create consolidated Pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })
        
        # ROC Data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, roc_auc)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    
    # Finalize Consolidated ROC Curve Plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', pad=20, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    roc_path = os.path.join(vis_dir, 'roc_curves_comparison.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"\n Saved overall ROC curves comparison to '{roc_path}'")
    
    # Display Result Summary Table
    results_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(results_df.to_string())
    print("=" * 70 + "\n")
    
    return fitted_pipelines, results_df

def plot_feature_importance(fitted_pipelines, preprocessor_transformer, numeric_features, categorical_features, vis_dir='visualizations'):
    """
    Extracts and visualizes the feature importances from the best ensemble model.
    """
    print("=" * 70)
    print("STEP 4: Model Interpretation & Feature Importance")
    print("=" * 70)
    
    # Best model selection (Gradient Boosting generally excels)
    best_model_name = "Gradient Boosting"
    pipeline = fitted_pipelines[best_model_name]
    
    # Get feature names after one-hot encoding
    ohe_categories = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_categories)
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, legend=False, palette="viridis")
    plt.title(f"Feature Importance Extraction ({best_model_name})", pad=20, fontweight='bold')
    plt.xlabel("Relative Importance Weight", fontweight='bold')
    plt.ylabel("Engineered & Base Features", fontweight='bold')
    plt.tight_layout()
    
    feat_path = os.path.join(vis_dir, 'feature_importance.png')
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f" Saved extracted feature importances plot to '{feat_path}'")
    print("\nTop 5 Most Powerful Drivers of Creditworthiness:")
    for i in range(5):
        print(f"  {i+1}. {sorted_features[i]}: {sorted_importances[i]:.4f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, preprocessor, num_feats, cat_feats = preprocess_and_engineer_features(df)
    pipelines, metrics_df = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    plot_feature_importance(pipelines, preprocessor, num_feats, cat_feats)
    print(">> Pipeline Execution Complete! All source models trained and visualizations rendered successfully.")
