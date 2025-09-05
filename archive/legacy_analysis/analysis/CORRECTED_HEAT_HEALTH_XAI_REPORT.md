# Heat-Health XAI Analysis Report (Corrected)

**Generated**: 2025-09-01 16:04:18

## Dataset Summary

- **Total records**: 1,239
- **Unique participants**: 1,239
- **Study period**: 2011-02-10 00:00:00 to 2018-07-24 00:00:00
- **Geographic scope**: Johannesburg metropolitan area
- **Data quality fixes applied**: Glucose unit conversion, outlier handling, impossible value correction

### Sample Composition

- **dphru_053**: 992 records (992 unique participants)
- **dphru_013**: 247 records (247 unique participants)

## Analysis Results by Biological Pathway

## Statistical Methodology

### Data Quality Assurance
- Unit standardization: Glucose converted from mmol/L to mg/dL
- Outlier handling: Extreme values (>3×IQR) capped
- Missing data: Complete case analysis for each pathway
- Impossible values: Set to NaN (negative cholesterol, extreme creatinine)

### Model Validation
- Cross-validation: 5-fold CV to prevent overfitting
- Train-test split: 80-20 stratified split
- Regularization: Max depth limits, minimum split size for tree models; L1/L2 for ElasticNet
- Feature selection: Excluded target-related variables to prevent leakage

### XAI Analysis
- SHAP values computed for models with R² > 0.01
- Sample size: Up to 1,000 observations for computational efficiency
- Hypothesis generation: Based on top 10 most important features

### Climate Data Integration
- Temperature data: ERA5 reanalysis + local weather stations
- Spatial resolution: Johannesburg metropolitan area
- Extreme heat definition: >95th percentile (26.8°C)
- Lag windows: 1, 3, 7, 14, 21, 28, 30, 60, 90 days
- Heat stress metrics: Daily heat stress and extreme heat days

