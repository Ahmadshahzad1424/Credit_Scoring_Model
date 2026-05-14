import numpy as np
import pandas as pd

def generate_credit_dataset(num_samples=5000, random_seed=42, output_path='credit_scoring_dataset.csv'):
    """
    Generates a realistic synthetic credit scoring dataset based on financial history.
    Includes numerical features, categorical features, class imbalances, and non-linear interactions.
    """
    np.random.seed(random_seed)
    
    print(f"Generating {num_samples} records for credit scoring analysis...")
    
    # 1. Base demographics & income
    customer_ids = [f"CUST_{100000 + i}" for i in range(num_samples)]
    age = np.random.randint(21, 70, size=num_samples)
    
    # Skewed income distribution mimicking real world scenarios
    annual_income = np.random.exponential(scale=45000, size=num_samples) + 15000
    annual_income = np.round(annual_income, 2)
    
    # Employment length correlated slightly with age
    employment_length = np.maximum(0, age - 20) * np.random.uniform(0.1, 0.8, size=num_samples)
    employment_length = np.round(employment_length, 1)
    
    # 2. Financial liabilities & usage
    # Total debt outstanding
    total_debt = annual_income * np.random.uniform(0.05, 0.85, size=num_samples)
    total_debt = np.round(total_debt, 2)
    
    # Debt-to-Income (DTI) ratio
    dti_ratio = np.round(total_debt / annual_income, 4)
    
    # Credit utilization ratio (percentage of available revolving credit used)
    credit_utilization = np.random.beta(a=2, b=5, size=num_samples)
    # Inject high utilization for a subset
    high_util_mask = np.random.rand(num_samples) < 0.25
    credit_utilization[high_util_mask] = np.random.uniform(0.7, 1.0, size=np.sum(high_util_mask))
    credit_utilization = np.round(credit_utilization, 4)
    
    # Number of open credit accounts
    num_open_accounts = np.random.poisson(lam=6, size=num_samples) + 1
    
    # 3. Payment history behavior
    # Missed payments history (0: flawless, >0: past delinquencies)
    missed_payments_probs = [0.65, 0.20, 0.10, 0.05]
    missed_payments = np.random.choice([0, 1, 2, 3], size=num_samples, p=missed_payments_probs)
    
    # 4. Current loan request details
    loan_amount = annual_income * np.random.uniform(0.1, 1.2, size=num_samples)
    loan_amount = np.round(loan_amount, 2)
    
    loan_purposes = ['Home', 'Car', 'Education', 'Personal', 'Business']
    loan_purpose = np.random.choice(loan_purposes, size=num_samples, p=[0.3, 0.25, 0.15, 0.2, 0.1])
    
    # 5. Formulate logical Target Variable: creditworthiness_target (1 = Creditworthy/Approved, 0 = High Risk/Default)
    # We construct a latent risk score combining positive and negative factors
    # Higher score -> more risk of default
    base_risk = (
        (dti_ratio * 2.5) + 
        (credit_utilization * 3.0) + 
        (missed_payments * 1.5) - 
        (np.log1p(annual_income) * 0.3) - 
        (employment_length * 0.05) +
        (loan_amount / annual_income * 1.2)
    )
    
    # Purpose modifier
    purpose_risk_map = {'Business': 0.5, 'Personal': 0.3, 'Education': 0.0, 'Car': -0.2, 'Home': -0.4}
    purpose_risk = np.array([purpose_risk_map[p] for p in loan_purpose])
    
    total_risk = base_risk + purpose_risk
    
    # Add Gaussian noise to make classification realistic but robust
    total_risk += np.random.normal(loc=0, scale=0.8, size=num_samples)
    
    # Thresholding to achieve an approximate 70/30 class balance (1: Good credit, 0: High risk)
    threshold = np.percentile(total_risk, 30)
    creditworthiness_target = (total_risk < threshold).astype(int)
    # Let's invert so 1 represents positive creditworthiness (approved) and 0 represents risky
    # Wait, if total_risk < threshold, those are the lowest risk customers, so they are creditworthy (1).
    # Let's check distribution: bottom 30% risk would mean 30% Approved, 70% Default. Let's invert the percentile to approve ~72% of customers.
    threshold = np.percentile(total_risk, 72)
    creditworthiness_target = (total_risk <= threshold).astype(int)
    
    # Build DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'annual_income': annual_income,
        'employment_length_years': employment_length,
        'total_debt_outstanding': total_debt,
        'debt_to_income_ratio': dti_ratio,
        'credit_utilization_ratio': credit_utilization,
        'num_open_accounts': num_open_accounts,
        'missed_payments_history': missed_payments,
        'loan_amount_requested': loan_amount,
        'loan_purpose': loan_purpose,
        'creditworthiness_target': creditworthiness_target
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully and saved to '{output_path}'.")
    print("\nTarget Class Distribution:")
    print(df['creditworthiness_target'].value_counts(normalize=True).rename({1: '1 (Creditworthy)', 0: '0 (High Risk)'}))
    return df

if __name__ == "__main__":
    generate_credit_dataset()
