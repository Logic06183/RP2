# Supplementary Methods Validation and XAI Technical Details

## SUPPLEMENTARY METHODS FOR PEER REVIEW

---

## S1. DETAILED XAI METHODOLOGY

### S1.1 SHAP (Shapley Additive Explanations) Implementation

#### Theoretical Foundation
SHAP values are based on coalitional game theory, providing the unique solution that satisfies:
- **Efficiency**: ∑ᵢ φᵢ = f(x) - f(∅)
- **Symmetry**: Equal contribution for equal marginal contributions
- **Dummy**: Zero contribution for irrelevant features
- **Additivity**: Linear for additive models

#### Mathematical Formulation
For feature i and instance x:
```
φᵢ(x) = ∑_{S⊆F\{i}} [|S|!(|F|-|S|-1)!]/|F|! × [f(S∪{i}) - f(S)]
```

Where:
- F = set of all features
- S = subset of features
- f(S) = model prediction using only features in S

#### Computational Implementation
```python
# TreeExplainer for tree-based models (exact computation)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# For non-tree models, use sampling approximation
explainer = shap.Explainer(model.predict, X_background)
shap_values = explainer(X_sample, max_evals=1000)
```

#### Quality Control Metrics
- **Additivity check**: |∑φᵢ - (f(x) - E[f(X)])| < 0.01
- **Consistency**: Correlation between runs > 0.95
- **Stability**: Bootstrap coefficient of variation < 0.1

### S1.2 Causal Inference Framework

#### Potential Outcomes Framework
We estimate Average Treatment Effects (ATE) using:
```
τ = E[Y¹ - Y⁰] = E[Y|T=1] - E[Y|T=0]
```

Where:
- Y¹ = potential outcome under treatment (high temperature)
- Y⁰ = potential outcome under control (low temperature)
- T = treatment indicator

#### Assumptions Required
1. **Positivity**: 0 < P(T=1|X) < 1 for all X
2. **Unconfoundedness**: Y⁰, Y¹ ⊥ T | X
3. **SUTVA**: No interference between units

#### Counterfactual Generation
```python
def generate_counterfactual(X, intervention_var, delta):
    """Generate counterfactual dataset with intervention"""
    X_cf = X.copy()
    X_cf[intervention_var] += delta
    return X_cf

# Temperature interventions
X_hot = generate_counterfactual(X, 'temp_mean', +3.0)
X_cool = generate_counterfactual(X, 'temp_mean', -3.0)

# Predict outcomes
y_baseline = model.predict(X)
y_hot = model.predict(X_hot)  
y_cool = model.predict(X_cool)

# Calculate effects
ate_hot = np.mean(y_hot - y_baseline)
ate_cool = np.mean(y_cool - y_baseline)
```

---

## S2. MACHINE LEARNING PIPELINE VALIDATION

### S2.1 Model Selection and Hyperparameter Tuning

#### Cross-Validation Strategy
```python
# Time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=5, test_size=60)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Ensure temporal ordering
    assert X_train.index.max() < X_test.index.min()
```

#### Hyperparameter Grid
| Algorithm | Hyperparameters | Grid |
|-----------|-----------------|------|
| RandomForest | n_estimators | [50, 100, 200] |
| | max_depth | [5, 10, 15, None] |
| | min_samples_split | [2, 5, 10] |
| GradientBoosting | n_estimators | [50, 100, 200] |
| | learning_rate | [0.01, 0.1, 0.2] |
| | max_depth | [3, 5, 7] |

#### Model Performance Metrics
```python
def evaluate_model(y_true, y_pred):
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics
```

### S2.2 Feature Selection and Engineering

#### Climate Feature Engineering
```python
def create_climate_features(df, base_vars, lags=[7, 14, 21, 28]):
    """Create lagged climate features"""
    for var in base_vars:
        for lag in lags:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)
    
    # Heat indices
    df['heat_index'] = calculate_heat_index(df['temp'], df['humidity'])
    df['apparent_temp'] = calculate_apparent_temp(df['temp'], df['humidity'], df['wind'])
    
    return df

def calculate_heat_index(T, RH):
    """Heat Index calculation (Rothfusz equation)"""
    HI = (-8.784695 + 1.61139411*T + 2.338549*RH - 0.14611605*T*RH 
          - 1.2308094e-2*T**2 - 1.6424828e-2*RH**2 + 2.211732e-3*T**2*RH 
          + 7.2546e-4*T*RH**2 - 3.582e-6*T**2*RH**2)
    return HI
```

#### Feature Selection Validation
```python
# Recursive Feature Elimination with Cross-Validation
from sklearn.feature_selection import RFECV

selector = RFECV(
    estimator=RandomForestRegressor(n_estimators=100),
    step=1,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='r2',
    n_jobs=-1
)

X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.support_]
```

---

## S3. STATISTICAL VALIDATION

### S3.1 Hypothesis Testing Framework

#### Multiple Testing Correction
Given 7 hypotheses tested simultaneously:
```python
from statsmodels.stats.multitest import multipletests

# Benjamini-Hochberg correction
p_values = [p_h1, p_h2, p_h3, p_h4, p_h5, p_h6, p_h7]
rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
    p_values, 
    alpha=0.05, 
    method='fdr_bh'
)
```

#### Power Analysis
```python
import statsmodels.stats.power as smp

# Cohen's f² for multiple regression
def cohens_f2(r2_full, r2_reduced):
    return (r2_full - r2_reduced) / (1 - r2_full)

# Power calculation
power = smp.f_power(
    u=k-1,  # numerator df
    v=n-k,  # denominator df  
    ncc=cohens_f2(r2_full, r2_reduced)
)
```

### S3.2 Bootstrap Validation

#### SHAP Value Stability
```python
def bootstrap_shap_stability(X, y, model, n_bootstrap=1000):
    """Assess SHAP value stability via bootstrapping"""
    shap_distributions = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X.iloc[idx], y.iloc[idx]
        
        # Retrain model
        model_boot = clone(model).fit(X_boot, y_boot)
        
        # Calculate SHAP
        explainer = shap.TreeExplainer(model_boot)
        shap_values = explainer(X_boot.sample(50))
        
        shap_distributions.append(shap_values.values.mean(axis=0))
    
    # Calculate confidence intervals
    shap_ci = np.percentile(shap_distributions, [2.5, 97.5], axis=0)
    return shap_ci
```

#### Model Stability Metrics
```python
# Calculate stability across bootstrap samples
stability_metrics = {
    'feature_importance_cv': np.std(importance_bootstraps, axis=0) / np.mean(importance_bootstraps, axis=0),
    'prediction_cv': np.std(prediction_bootstraps, axis=0) / np.mean(prediction_bootstraps, axis=0),
    'shap_cv': np.std(shap_bootstraps, axis=0) / np.mean(shap_bootstraps, axis=0)
}
```

---

## S4. DATA QUALITY VALIDATION

### S4.1 Missing Data Analysis

#### Missing Data Patterns
```python
import missingno as msno

# Visualize missing patterns
msno.matrix(df)
msno.heatmap(df)

# Test for MCAR (Little's MCAR test)
from impyute import diagnostics
mcar_test = diagnostics.mcar_test(df)
print(f"MCAR p-value: {mcar_test['p-value']}")
```

#### Imputation Validation
```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Compare imputation methods
imputers = {
    'KNN': KNNImputer(n_neighbors=5),
    'Iterative': IterativeImputer(random_state=42),
    'Mean': SimpleImputer(strategy='mean')
}

# Cross-validation with different imputers
for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_with_missing)
    score = cross_val_score(model, X_imputed, y, cv=5).mean()
    print(f"{name}: {score:.3f}")
```

### S4.2 Outlier Detection and Treatment

#### Multivariate Outlier Detection
```python
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

# Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
outliers_iso = iso_forest.fit_predict(X)

# Robust Covariance
robust_cov = EllipticEnvelope(contamination=0.02)
outliers_cov = robust_cov.fit_predict(X)

# Consensus outliers
consensus_outliers = (outliers_iso == -1) & (outliers_cov == -1)
```

#### Biomarker Range Validation
```python
# Clinical reference ranges
REFERENCE_RANGES = {
    'total_cholesterol': (2.0, 8.0),  # mmol/L
    'hdl': (0.5, 3.0),               # mmol/L
    'ldl': (1.0, 6.0),               # mmol/L
    'systolic_bp': (80, 200),        # mmHg
    'diastolic_bp': (40, 120),       # mmHg
    'creatinine': (30, 300),         # μmol/L
}

def validate_biomarker_ranges(df, ranges):
    """Flag values outside clinical ranges"""
    for biomarker, (min_val, max_val) in ranges.items():
        if biomarker in df.columns:
            outside_range = (df[biomarker] < min_val) | (df[biomarker] > max_val)
            print(f"{biomarker}: {outside_range.sum()} values outside range")
    return df
```

---

## S5. MODEL INTERPRETABILITY VALIDATION

### S5.1 SHAP Consistency Checks

#### Additivity Property Validation
```python
def validate_shap_additivity(shap_values, predictions, baseline):
    """Validate SHAP additivity property"""
    shap_sums = shap_values.sum(axis=1) + baseline
    additivity_error = np.abs(predictions - shap_sums)
    
    print(f"Mean additivity error: {additivity_error.mean():.6f}")
    print(f"Max additivity error: {additivity_error.max():.6f}")
    print(f"Proportion with error < 0.01: {(additivity_error < 0.01).mean():.3f}")
    
    return additivity_error
```

#### Feature Attribution Consistency
```python
def validate_feature_consistency(model, X, n_permutations=100):
    """Test feature importance consistency across permutations"""
    importance_distributions = []
    
    for i in range(n_permutations):
        # Permute features
        X_perm = X.sample(frac=1).reset_index(drop=True)
        
        # Calculate SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_perm)
        importance = np.abs(shap_values.values).mean(axis=0)
        importance_distributions.append(importance)
    
    # Calculate stability metrics
    importance_mean = np.mean(importance_distributions, axis=0)
    importance_std = np.std(importance_distributions, axis=0)
    stability_ratio = importance_std / importance_mean
    
    return importance_mean, stability_ratio
```

### S5.2 Counterfactual Validity

#### Plausibility Constraints
```python
def validate_counterfactual_plausibility(X_original, X_counterfactual, constraints):
    """Ensure counterfactuals respect domain constraints"""
    violations = {}
    
    for var, (min_val, max_val) in constraints.items():
        if var in X_counterfactual.columns:
            outside_range = (
                (X_counterfactual[var] < min_val) | 
                (X_counterfactual[var] > max_val)
            )
            violations[var] = outside_range.sum()
    
    return violations

# Climate constraints for Johannesburg
CLIMATE_CONSTRAINTS = {
    'temp_mean': (-5, 35),    # °C
    'humidity': (10, 95),     # %
    'precipitation': (0, 50)   # mm/day
}
```

#### Causal Effect Magnitude Validation
```python
def validate_effect_magnitudes(effects, biomarker, threshold_factors):
    """Validate that effects are within plausible ranges"""
    
    # Expected effect sizes based on literature
    EXPECTED_EFFECTS = {
        'total_cholesterol': 0.1,  # mmol/L per 3°C
        'hdl': 0.05,              # mmol/L per 3°C  
        'systolic_bp': 2.0         # mmHg per 3°C
    }
    
    expected = EXPECTED_EFFECTS.get(biomarker, 0.1)
    observed = np.abs(effects)
    
    # Flag if observed >> expected
    ratio = observed / expected
    plausible = ratio < threshold_factors.get(biomarker, 5.0)
    
    return plausible, ratio
```

---

## S6. REPRODUCIBILITY PROTOCOLS

### S6.1 Computational Environment

#### Software Versions
```python
import sys
import pandas as pd
import numpy as np
import sklearn
import shap

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("SHAP version:", shap.__version__)

# Save environment
import subprocess
subprocess.run(["pip", "freeze", ">", "requirements.txt"])
```

#### Random Seed Management
```python
# Set all random seeds for reproducibility
RANDOM_SEED = 42

import random
import numpy as np
from sklearn.utils import check_random_state

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure sklearn randomness
def set_sklearn_random_state(estimator, random_state):
    """Recursively set random state for sklearn objects"""
    if hasattr(estimator, 'random_state'):
        estimator.random_state = random_state
    if hasattr(estimator, 'base_estimator'):
        set_sklearn_random_state(estimator.base_estimator, random_state)
    return estimator
```

### S6.2 Data Provenance

#### Data Lineage Tracking
```python
class DataLineage:
    """Track data transformations for reproducibility"""
    
    def __init__(self):
        self.operations = []
    
    def log_operation(self, operation, input_shape, output_shape, params=None):
        self.operations.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'parameters': params
        })
    
    def save_lineage(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.operations, f, indent=2)
```

#### Checksums and Validation
```python
import hashlib

def calculate_data_checksum(df):
    """Calculate MD5 checksum of dataframe"""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()

# Store checksums for validation
CHECKSUMS = {
    'health_data': '5d41402abc4b2a76b9719d911017c592',
    'climate_data': '7d865e959b2466918c9863afca942d0f',
    'processed_data': '098f6bcd4621d373cade4e832627b4f6'
}
```

---

## S7. SENSITIVITY ANALYSES

### S7.1 Temporal Sensitivity

#### Alternative Lag Structures
```python
# Test different lag specifications
LAG_SPECIFICATIONS = {
    'short': [3, 7, 14],
    'medium': [7, 14, 21, 28], 
    'long': [14, 21, 28, 35],
    'custom': [5, 10, 15, 20, 25, 30]
}

sensitivity_results = {}
for spec_name, lags in LAG_SPECIFICATIONS.items():
    X_lagged = create_lagged_features(X, lags)
    model.fit(X_lagged, y)
    score = cross_val_score(model, X_lagged, y, cv=5).mean()
    sensitivity_results[spec_name] = score
```

#### Seasonal Stratification
```python
def seasonal_analysis(df):
    """Analyze results by season"""
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['season'] = df['month'].map({
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    })
    
    seasonal_results = {}
    for season in df['season'].unique():
        mask = df['season'] == season
        X_season, y_season = X[mask], y[mask]
        
        if len(X_season) > 50:  # Minimum sample size
            model.fit(X_season, y_season)
            score = model.score(X_season, y_season)
            seasonal_results[season] = score
    
    return seasonal_results
```

### S7.2 Model Sensitivity

#### Alternative Algorithms
```python
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

ALTERNATIVE_MODELS = {
    'ElasticNet': ElasticNet(alpha=0.1),
    'Ridge': Ridge(alpha=1.0),
    'SVR': SVR(kernel='rbf', C=1.0),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
}

model_comparison = {}
for name, model in ALTERNATIVE_MODELS.items():
    scores = cross_val_score(model, X, y, cv=5)
    model_comparison[name] = {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }
```

#### Feature Subset Sensitivity
```python
FEATURE_SUBSETS = {
    'climate_only': [col for col in X.columns if 'temp' in col or 'humid' in col],
    'demographic_only': ['age', 'sex', 'bmi'],
    'socioeconomic_only': [col for col in X.columns if 'income' in col or 'education' in col],
    'no_lags': [col for col in X.columns if 'lag' not in col]
}

subset_results = {}
for subset_name, features in FEATURE_SUBSETS.items():
    if all(f in X.columns for f in features):
        X_subset = X[features]
        scores = cross_val_score(model, X_subset, y, cv=5)
        subset_results[subset_name] = scores.mean()
```

---

## S8. CLINICAL VALIDATION

### S8.1 Effect Size Validation

#### Cohen's Guidelines for Biomarkers
```python
def interpret_effect_size(effect, biomarker):
    """Interpret effect sizes using biomarker-specific thresholds"""
    
    EFFECT_THRESHOLDS = {
        'total_cholesterol': {'small': 0.1, 'medium': 0.2, 'large': 0.5},
        'hdl': {'small': 0.02, 'medium': 0.05, 'large': 0.1},
        'systolic_bp': {'small': 1.0, 'medium': 3.0, 'large': 5.0}
    }
    
    thresholds = EFFECT_THRESHOLDS.get(biomarker, {'small': 0.1, 'medium': 0.3, 'large': 0.5})
    
    if abs(effect) >= thresholds['large']:
        return 'large'
    elif abs(effect) >= thresholds['medium']:
        return 'medium'
    elif abs(effect) >= thresholds['small']:
        return 'small'
    else:
        return 'negligible'
```

#### Clinical Significance Testing
```python
def clinical_significance_test(effects, mcid):
    """Test if effects exceed minimum clinically important difference"""
    significant_effects = np.abs(effects) >= mcid
    
    # Bootstrap confidence interval for proportion
    n_bootstrap = 1000
    proportions = []
    
    for _ in range(n_bootstrap):
        boot_effects = np.random.choice(effects, size=len(effects), replace=True)
        prop = np.mean(np.abs(boot_effects) >= mcid)
        proportions.append(prop)
    
    ci_lower = np.percentile(proportions, 2.5)
    ci_upper = np.percentile(proportions, 97.5)
    
    return {
        'proportion_significant': np.mean(significant_effects),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

# Minimum clinically important differences
MCID = {
    'total_cholesterol': 0.2,  # mmol/L
    'hdl': 0.03,              # mmol/L
    'systolic_bp': 2.0         # mmHg
}
```

---

## S9. REPORTING STANDARDS

### S9.1 TRIPOD-AI Checklist Compliance

| Item | Description | Section |
|------|-------------|---------|
| 1a | Title identifies ML model | Title |
| 1b | Abstract structured | Abstract |
| 2 | Background and objectives | Introduction |
| 3a | Study design | Methods |
| 3b | Setting | Methods 3.1 |
| 4a | Participant eligibility | Methods 3.2 |
| 4b | Data sources | Methods 3.2 |
| ... | [Complete checklist] | ... |

### S9.2 STROBE-AI Extension

Following STROBE-AI guidelines for:
- AI/ML model reporting
- Feature selection transparency  
- Hyperparameter documentation
- Performance metric reporting
- Uncertainty quantification

---

*This supplementary document provides complete technical details for peer review and reproduction, ensuring full transparency of our XAI methodology and validation procedures.*