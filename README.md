
## Project Objective

Given attributes about a loan applicant, decide the credibility of the borrower with the Probability of Default (PD) and financial stress score and justify the decision using XAI techniques.

**Dataset:** German Credit Data (UCI Machine Learning Repository)  
<!-- **Problem type:** Binary Classification   -->
**Stack:** Python, Pandas, Plotly, Scikit-learn, XGBoost, SHAP, Streamlit, MLflow , Statsmodels, SciPy, Jupyter notebook


---

<!-- ##  Project Structure
credit-risk-project/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Engineered + split data
├── notebooks/
|   ├── mlruns/                 # mlflow
│   ├── 01_eda.ipynb            # EDA + hypothesis testing
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_mlops.ipynb
│   └── 05_explainability.ipynb
├── src/                        # Modular Python scripts
├── app/
│   └── streamlit_app.py        # Deployed Streamlit app
├── models/
│   └── calibrated_pipeline.pkl
├── reports/figures/            # Key plots
├── README.md                   # project description
└── requirements.txt
--- -->

## Run Locally
```bash
# Clone repo
git clone https://github.com/Abhinav289/credit-risk-project.git
cd credit-risk-project

# Create environment
conda create -n credit_risk python=3.11
conda activate credit_risk

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd app
streamlit run streamlit_app.py
```

---

### Key EDA Findings

### Default Borrower Profile
A borrower most likely to default shows:

| Feature | High Risk Signal |
|---|---|
| Checking account | Negative or absent balance |
| Credit history | No credits / all paid |
| Savings | Less than 100 DM |
| Property | No known collateral |
| Other parties | Co-applicant |
| Duration | 45+ months |
| Age | 25–30 years | 

### Hypothesis Testing
All features validated with statistical tests (α = 0.05):
- **Mann-Whitney U** for numeric features
- **Chi-Square** for categorical features
- **Wilcoxon Signed-Rank** for model comparison

## Feature Engineering

3 domain-driven features engineered:

| Feature | Logic | Impact |
|---|---|---|
| `financial_stress_score` | checking + savings stress combined | Score 0→6: 8.87%→52.05% default rate |
| `monthly_burden` | credit_amount / duration | Captures repayment affordability |
| `high_risk_interaction` | negative checking AND no credit history | 69.12% default rate when flagged |

--- 
## Design Patterns Applied

| Pattern | Implementation |
|---|---|
| **Rebalancing** | SMOTE applied on training set only — never on test |
| **Reframing** | Probability calibration via CalibratedClassifierCV |
| **Checkpoints** | XGBoost early stopping (stopped at round 107)|
| **Explainable Predictions** | SHAP waterfall per individual prediction |
| **Repeatable Splitting** | Stratified train-test split |

---

### Experiment tracked with MLflow:

| Parameter | Value |
|---|---|
| Model | XGBoost + StandardScaler |
| n_estimators | 107 (early stopping) |
| max_depth | 4 |
| learning_rate | 0.1 |
| Calibration | Isotonic (cv=5) |
| Threshold | 0.612 |

---

## Performance

| Model | CV ROC-AUC | F1 (threshold=0.612) |
|---|---|---|
| Logistic Regression (baseline) | 0.856 ± 0.013 | 0.55 |
| **XGBoost** | **0.890 ± 0.019** | **0.614** |

**Statistical comparison:** Wilcoxon Signed-Rank test (p=0.0625)  
XGBoost outperformed LR on every single fold.  
Threshold optimized at 0.612 to maximize F1 score.

---
## References

- Hofmann, H. (1994). German Credit Data. UCI ML Repository
- Lakshmanan et al. — Machine Learning Design Patterns (O'Reilly)
<!-- - Naeem Siddiqi — Credit Risk Scorecards -->
- Demšar, J. (2006). Statistical Comparisons of Classifiers
<!-- - SHAP: Lundberg & Lee (2017). A Unified Approach to  
  Interpreting Model Predictions -->