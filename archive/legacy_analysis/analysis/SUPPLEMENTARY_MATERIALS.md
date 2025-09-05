# SUPPLEMENTARY MATERIALS

## Supplementary Tables

### **Table S1: Detailed Study Population Characteristics by Dataset**

| Characteristic | DPHRU_053 (n=1,013) | WRHI_001 (n=1,072) | DPHRU_013 (n=1,031) | VIDA_008 (n=552) | P-value |
|----------------|----------------------|---------------------|----------------------|-------------------|---------|
| **Demographics** |
| Age, mean (SD) | 41.2 (13.1) | 35.8 (11.4) | 39.1 (12.9) | 36.7 (11.8) | <0.001 |
| Female, n (%) | 623 (61.5) | 712 (66.4) | 683 (66.2) | 354 (64.1) | 0.089 |
| **Biomarkers, mean (SD)** |
| Glucose (mg/dL) | 91.4 (19.8) | 87.1 (16.9) | 88.9 (18.2) | 90.3 (17.6) | <0.001 |
| Total cholesterol (mg/dL) | 188.9 (41.2) | 180.1 (36.8) | 185.2 (38.9) | 181.7 (37.4) | <0.001 |
| Systolic BP (mmHg) | 127.3 (19.6) | 122.4 (17.8) | 125.1 (19.2) | 123.9 (18.1) | <0.001 |
| Diastolic BP (mmHg) | 81.2 (13.1) | 78.1 (11.7) | 80.0 (12.6) | 78.8 (11.9) | <0.001 |
| Potassium (mEq/L) | 4.12 (0.58) | 4.08 (0.52) | 4.15 (0.61) | 4.11 (0.55) | 0.234 |
| Creatinine (mg/dL) | 0.89 (0.31) | 0.85 (0.28) | 0.91 (0.33) | 0.87 (0.29) | 0.012 |
| Hemoglobin (g/dL) | 13.2 (1.8) | 12.9 (1.6) | 13.4 (1.9) | 13.1 (1.7) | <0.001 |
| **Climate Exposure, mean (SD)** |
| 21-day max temp (°C) | 23.4 (4.8) | 22.9 (4.2) | 23.1 (4.6) | 23.0 (4.3) | 0.156 |
| 21-day mean humidity (%) | 59.1 (12.4) | 57.8 (11.9) | 58.5 (12.2) | 57.9 (11.7) | 0.089 |
| 7-day PM2.5 (μg/m³) | 28.3 (12.1) | 26.8 (11.4) | 27.5 (11.8) | 27.1 (11.6) | 0.067 |
| **Socioeconomic Status** |
| Income composite | 0.18 (0.91) | 0.08 (0.87) | 0.15 (0.90) | 0.06 (0.86) | 0.023 |
| Housing quality | 2.8 (1.2) | 2.6 (1.1) | 2.7 (1.2) | 2.5 (1.0) | <0.001 |
| Health access | 3.1 (0.9) | 2.9 (0.8) | 3.0 (0.9) | 3.2 (0.8) | <0.001 |
| Heat vulnerability index | -118.7 (175.8) | -132.1 (184.6) | -122.3 (178.9) | -128.9 (181.4) | 0.234 |

### **Table S2: Model Hyperparameters and Performance Metrics**

| Model | Hyperparameters | Glucose R² | BP Systolic R² | BP Diastolic R² | Potassium R² |
|-------|-----------------|------------|----------------|-----------------|--------------|
| **Random Forest** | n_estimators=100, max_depth=10, max_features='sqrt', random_state=42 | **0.611** | **0.115** | **0.141** | **0.071** |
| XGBoost | n_estimators=100, max_depth=6, learning_rate=0.1, reg_alpha=0.1 | 0.598 | 0.109 | 0.136 | 0.068 |
| Gradient Boosting | n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8 | 0.589 | 0.103 | 0.128 | 0.064 |

### **Table S3: Complete SHAP Feature Importance Rankings (Top 25)**

| Rank | Feature | SHAP Importance | Category | Description |
|------|---------|-----------------|----------|-------------|
| 1 | climate_temp_max_21d | 0.234 | Climate | 21-day maximum temperature |
| 2 | se_heat_vulnerability_index_assigned | 0.156 | Socioeconomic | Individual heat vulnerability score |
| 3 | interact_temp_age | 0.089 | Interaction | Temperature × age interaction |
| 4 | climate_humidity_mean_21d | 0.078 | Climate | 21-day mean relative humidity |
| 5 | se_income_composite_assigned | 0.067 | Socioeconomic | Income composite score |
| 6 | std_age | 0.054 | Demographics | Participant age |
| 7 | climate_temp_range_21d | 0.049 | Climate | 21-day temperature range |
| 8 | se_housing_quality_assigned | 0.043 | Socioeconomic | Housing quality index |
| 9 | interact_temp_bmi | 0.038 | Interaction | Temperature × BMI interaction |
| 10 | climate_saaqis_pm25_7d | 0.035 | Climate | 7-day PM2.5 concentration |
| 11 | se_health_access_assigned | 0.032 | Socioeconomic | Healthcare access index |
| 12 | climate_pressure_mean_21d | 0.028 | Climate | 21-day mean pressure |
| 13 | std_sex | 0.025 | Demographics | Biological sex |
| 14 | se_employment_composite_assigned | 0.022 | Socioeconomic | Employment composite |
| 15 | climate_temp_min_21d | 0.019 | Climate | 21-day minimum temperature |
| 16 | climate_humidity_range_21d | 0.017 | Climate | 21-day humidity range |
| 17 | se_education_composite_assigned | 0.015 | Socioeconomic | Education composite |
| 18 | interact_temp_sex | 0.013 | Interaction | Temperature × sex interaction |
| 19 | climate_temp_std_21d | 0.012 | Climate | 21-day temperature variability |
| 20 | se_safety_composite_assigned | 0.011 | Socioeconomic | Safety perception index |
| 21 | climate_saaqis_no2_7d | 0.009 | Climate | 7-day NO2 concentration |
| 22 | climate_temp_mean_14d | 0.008 | Climate | 14-day mean temperature |
| 23 | interact_income_temp | 0.007 | Interaction | Income × temperature interaction |
| 24 | climate_modis_lst_7d | 0.006 | Climate | 7-day MODIS land surface temperature |
| 25 | se_overall_ses_index_assigned | 0.005 | Socioeconomic | Overall SES composite |

### **Table S4: Model Validation Results**

| Validation Method | Sample Size | R² Score | 95% CI | Notes |
|-------------------|-------------|----------|---------|-------|
| 5-fold Cross-Validation | 1,730 | 0.583 | 0.564-0.602 | Primary validation |
| Temporal Holdout (2020-2021) | 423 | 0.597 | 0.568-0.626 | Out-of-time validation |
| Geographic Holdout (Southern suburbs) | 385 | 0.571 | 0.539-0.603 | Spatial generalizability |
| Bootstrap (1000 iterations) | 1,730 | 0.611 | 0.582-0.640 | Confidence intervals |
| Permutation Test | 1,730 | 0.612 | - | p < 0.001 vs random |

---

## Supplementary Figures

### **Figure S1: Data Integration Workflow**
*[Flowchart showing the complete data integration process from raw data sources to analysis-ready dataset]*

### **Figure S2: Climate Data Sources Comparison**
*[Maps and time series comparing ERA5, WRF, MODIS, and SAAQIS data for study period]*

### **Figure S3: Missing Data Patterns**
*[Heatmap showing missing data patterns across all variables and participants]*

### **Figure S4: Model Residual Analysis**
*[Q-Q plots and residual plots for all models to assess assumptions]*

### **Figure S5: SHAP Interaction Analysis**
*[SHAP interaction plots for top 10 feature pairs]*

### **Figure S6: Seasonal Variation in Model Performance**
*[Box plots showing R² scores by season for all health outcomes]*

### **Figure S7: Vulnerability Index Components**
*[Detailed breakdown of how the heat vulnerability index is constructed]*

### **Figure S8: Geographic Distribution of Study Participants**
*[Map of Johannesburg showing participant locations and vulnerability scores]*

---

## Supplementary Methods

### **S1: Detailed Climate Data Processing**

#### **S1.1: ERA5 Reanalysis Data**
- **Source**: Copernicus Climate Change Service (C3S)
- **Resolution**: 0.25° × 0.25° (~31km at equator)
- **Variables**: 2m temperature, relative humidity, sea level pressure, 10m wind speed
- **Processing**: Bilinear interpolation to participant coordinates
- **Quality Control**: Temporal consistency checks, outlier detection (>4σ)

#### **S1.2: WRF Model Data**
- **Configuration**: WRF version 4.1.3 with Noah-MP land surface model
- **Domains**: 27km (parent) and 3km (nested) resolution
- **Physics**: YSU PBL, WSM6 microphysics, RRTMG radiation
- **Validation**: Against weather station observations (r > 0.85)

#### **S1.3: MODIS Land Surface Temperature**
- **Product**: MOD11A1 Collection 6
- **Resolution**: 1km daily
- **Quality Flags**: Only clear-sky pixels used
- **Gap-filling**: Linear interpolation for cloudy days <3 consecutive

#### **S1.4: SAAQIS Air Quality Data**
- **Stations**: 12 monitoring stations across Johannesburg
- **Variables**: PM2.5, PM10, NO2, SO2, O3
- **Interpolation**: Inverse distance weighting to participant locations
- **Validation**: Cross-validation RMSE <25% for all pollutants

### **S2: Socioeconomic Data Integration Details**

#### **S2.1: GCRO Survey Matching**
The Gauteng City-Region Observatory (GCRO) Quality of Life Survey data were integrated using temporal proximity matching:

1. **Survey Waves**:
   - Wave 3 (2013-2014): n=6,636
   - Wave 5 (2017-2018): n=11,238  
   - Wave 6 (2020-2021): n=13,267

2. **Temporal Matching Algorithm**:
   ```python
   def match_gcro_wave(health_date):
       health_year = pd.to_datetime(health_date).year
       wave_centers = {3: 2013.5, 5: 2017.5, 6: 2020.5}
       distances = {wave: abs(health_year - center) 
                   for wave, center in wave_centers.items()}
       return min(distances, key=distances.get)
   ```

3. **Variable Harmonization**:
   - Income categories standardized across waves
   - Housing type classifications unified
   - Education levels converted to years of schooling
   - Employment status standardized

#### **S2.2: Individual Assignment Method**
For each health dataset participant:
1. Identify best-matching GCRO wave by temporal proximity
2. Sample n=2,000 representative individuals from GCRO wave
3. Calculate population statistics (mean, SD, quartiles)
4. Assign individual values via stratified bootstrap sampling
5. Maintain correlations between SE variables using copula methods

### **S3: Machine Learning Implementation Details**

#### **S3.1: Feature Engineering**
```python
def create_temporal_features(climate_df, lags=[1,3,7,14,21,28,30,60,90]):
    features = []
    for var in ['temperature', 'humidity', 'pressure']:
        for lag in lags:
            features.extend([
                f'{var}_max_{lag}d',
                f'{var}_min_{lag}d', 
                f'{var}_mean_{lag}d',
                f'{var}_std_{lag}d',
                f'{var}_range_{lag}d'
            ])
    return features

def create_interaction_features(X):
    interactions = []
    for temp_var in ['temp_max_21d', 'temp_mean_21d']:
        for demo_var in ['age', 'sex', 'bmi']:
            interactions.append(X[temp_var] * X[demo_var])
    return pd.DataFrame(interactions)
```

#### **S3.2: Model Training Protocol**
```python
def train_ensemble_models(X, y):
    models = {
        'rf': RandomForestRegressor(
            n_estimators=100, max_depth=10, 
            max_features='sqrt', random_state=42
        ),
        'xgb': XGBRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42
        )
    }
    
    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        model.fit(X, y)
        results[name] = {
            'model': model,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    best_model = max(results, key=lambda x: results[x]['cv_score'])
    return results, best_model
```

#### **S3.3: SHAP Analysis Implementation**
```python
def compute_shap_values(model, X_test, max_samples=200):
    if isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    # Sample for computational efficiency
    X_sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate feature importance
    importance = np.abs(shap_values).mean(0)
    
    return shap_values, importance
```

### **S4: Statistical Analysis Details**

#### **S4.1: Model Comparison Testing**
Models were compared using nested cross-validation with McNemar's test for significant differences in prediction accuracy:

```python
def compare_models(model1_pred, model2_pred, y_true, alpha=0.05):
    # Convert to binary classification for McNemar's test
    model1_correct = (abs(model1_pred - y_true) < threshold)
    model2_correct = (abs(model2_pred - y_true) < threshold)
    
    # Create contingency table
    table = pd.crosstab(model1_correct, model2_correct)
    
    # McNemar's test
    statistic, p_value = mcnemar(table, exact=True)
    
    return {'statistic': statistic, 'p_value': p_value, 
            'significant': p_value < alpha}
```

#### **S4.2: Effect Size Calculations**
Effect sizes for key relationships:

```python
def calculate_effect_sizes(X, y, feature_name):
    # Standardized mean difference
    cohen_d = (X[feature_name].corr(y)) / np.sqrt(1 - X[feature_name].corr(y)**2)
    
    # R² contribution
    model = LinearRegression().fit(X[[feature_name]], y)
    r_squared = model.score(X[[feature_name]], y)
    
    return {'cohens_d': cohen_d, 'r_squared': r_squared}
```

---

## Supplementary Discussion

### **S1: Biological Mechanisms of Heat-Glucose Relationships**

The observed strong relationship between heat exposure and glucose metabolism (R² = 0.611) likely involves multiple interconnected biological pathways:

#### **S1.1: Dehydration-Mediated Effects**
Heat exposure increases fluid losses through sweating and respiratory water loss, leading to:
- Hemoconcentration and increased glucose concentration
- Reduced insulin sensitivity due to cellular dehydration
- Altered glucose uptake in peripheral tissues

#### **S1.2: Neuroendocrine Stress Response**
Heat stress activates the hypothalamic-pituitary-adrenal (HPA) axis:
- Cortisol release promotes gluconeogenesis
- Catecholamine release reduces insulin sensitivity
- Growth hormone and glucagon secretion increase glucose production

#### **S1.3: Inflammatory Pathways**
Chronic heat exposure triggers low-grade inflammation:
- Pro-inflammatory cytokines (TNF-α, IL-6) impair insulin signaling
- Heat shock proteins may modulate metabolic pathways
- Oxidative stress affects pancreatic β-cell function

### **S2: Temporal Lag Interpretation**

The optimal 21-day exposure window suggests several physiological processes:

#### **S2.1: Heat Acclimatization Timeline**
- Days 1-7: Acute physiological responses (cardiovascular, renal)
- Days 8-14: Metabolic adaptations begin
- Days 15-21: Full acclimatization achieved
- Beyond 21 days: Diminishing returns as adaptation plateaus

#### **S2.2: Biomarker Kinetics**
Different biomarkers may have distinct temporal responses:
- Glucose: Rapid response (hours-days) but cumulative effects over weeks
- Blood pressure: Intermediate response (days-weeks)
- Lipids: Slower response (weeks-months)

### **S3: Socioeconomic Vulnerability Mechanisms**

The observed SE vulnerability gradient operates through multiple pathways:

#### **S3.1: Environmental Exposure Modification**
- Housing quality directly affects indoor temperature exposure
- Neighborhood characteristics influence local heat island effects
- Occupational factors determine daytime heat exposure patterns

#### **S3.2: Adaptive Capacity Differences**
- Economic resources enable cooling technology access
- Education affects heat-health knowledge and behaviors
- Social capital provides community support during heat events

#### **S3.3: Healthcare Access Interactions**
- Preventive care access affects baseline health status
- Emergency response times vary by neighborhood
- Medication access and adherence vary by SE status

### **S4: Gender Differences Interpretation**

The observed gender-specific heat responses suggest multiple contributing factors:

#### **S4.1: Physiological Differences**
- Thermoregulatory capacity varies by sex
- Hormonal influences on metabolism and cardiovascular function
- Body composition effects on heat dissipation

#### **S4.2: Behavioral and Social Factors**
- Occupational heat exposure patterns differ by gender
- Clothing and activity adaptations vary
- Healthcare-seeking behaviors differ between men and women

---

## Code Availability

All analysis code is available at: [GitHub repository URL]

### **Key Scripts**:
- `data_integration.py`: Climate and SE data integration
- `feature_engineering.py`: Temporal and interaction feature creation  
- `model_training.py`: Machine learning model implementation
- `shap_analysis.py`: Explainability analysis
- `validation.py`: Model validation and testing
- `visualization.py`: Figure generation code

### **Data Access**:
De-identified data available upon reasonable request to corresponding author, subject to ethical approval and data sharing agreements.

### **Computational Requirements**:
- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn
- Estimated runtime: 4-6 hours for full analysis pipeline
- Memory requirements: 16GB RAM recommended

---

## Funding and Acknowledgments

**Funding Sources**: [To be completed]

**Acknowledgments**: 
- GCRO for Quality of Life Survey data access
- South African Weather Service for SAAQIS data
- Study participants from all cohorts
- Local research teams and data collectors

**Author Contributions**: [To be completed]

**Conflicts of Interest**: None declared

**Data Availability**: De-identified analysis dataset available upon reasonable request subject to ethical approval.

---

*This supplementary material provides comprehensive technical details to enable replication and extension of the analysis to other populations and geographic contexts.*