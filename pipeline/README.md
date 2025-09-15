# HEAT Analysis Pipeline - Modular Framework

A clean, modular pipeline for climate-health analysis using machine learning and explainable AI.

## 🏗️ Pipeline Architecture

The pipeline is divided into three main modules:

### 1. Data Loading (`01_data_loader.py`)
- Loads and validates the master integrated dataset (128,465 records)
- Handles data type issues and mixed formats
- Provides comprehensive validation reporting
- Ensures data quality and completeness

### 2. Data Preprocessing (`02_data_preprocessor.py`)
- Feature engineering for climate variables
- Missing value handling with adaptive strategies
- Categorical variable encoding
- Standardization and scaling for ML

### 3. Machine Learning Analysis (`03_ml_analyzer.py`)
- Random Forest models for each biomarker
- Temporal cross-validation to prevent data leakage
- SHAP explainable AI analysis
- Comprehensive performance evaluation

## 🚀 Quick Start

### Run Complete Pipeline
```bash
cd pipeline/
python run_complete_pipeline.py
```

### Test Pipeline (Development Only)
```bash
python run_complete_pipeline.py --mode validate
```

### Run Individual Modules
```bash
# Data loading only
python 01_data_loader.py

# Preprocessing only
python 02_data_preprocessor.py

# ML analysis only
python 03_ml_analyzer.py
```

## 📊 Expected Results

The pipeline produces:

### Key Performance Metrics
- **CD4 Count**: R² = 0.60-0.70 (n = 1,300+)
- **Glucose**: R² = 0.55-0.65 (n = 2,700+)
- **Cholesterol**: R² = 0.50-0.60 (n = 3,000+)

### Output Files
- `../results/analysis_results.json` - Complete analysis results
- `../results/model_performance_summary.csv` - Performance table
- `../results/model_performance_summary.png` - Visualization

## 🔍 Data Validation

The pipeline validates:
- **128,465 total records** (119,362 GCRO + 9,103 Clinical)
- **Geographic coverage**: Johannesburg metropolitan area
- **Temporal coverage**: 2002-2021 (19 years)
- **Climate integration**: 78 ERA5 features
- **Biomarker completeness**: 95.7% data quality

## ⚠️ Critical Requirements

### Data Usage
- **NEVER sample the data** for publication analysis
- Always use `sample_fraction=1.0` in production
- Validate data scale: 128k+ records required
- Check temporal ordering for cross-validation

### Performance Standards
- Target R² > 0.5 for publication quality
- Use temporal cross-validation only
- Include SHAP explainability analysis
- Validate geographic and temporal coverage

## 🧪 Module Testing

Each module can be tested independently:

```python
# Test data loader
from data_loader import HeatDataLoader
loader = HeatDataLoader("../data")
df = loader.load_master_dataset(sample_fraction=1.0)
validation = loader.validate_data_structure()

# Test preprocessor
from data_preprocessor import HeatDataPreprocessor
preprocessor = HeatDataPreprocessor()
processed_df, report = preprocessor.preprocess_complete_pipeline(df)

# Test ML analyzer
from ml_analyzer import HeatMLAnalyzer
analyzer = HeatMLAnalyzer("../results")
results = analyzer.analyze_all_biomarkers(processed_df, targets, features)
```

## 📁 Output Structure

```
results/
├── analysis_results.json          # Complete results (JSON)
├── model_performance_summary.csv  # Performance table
├── model_performance_summary.png  # Summary plots
└── pipeline_execution.log         # Execution log
```

## 🔧 Configuration

### Model Parameters
- **Algorithm**: Random Forest (n_estimators=100)
- **Cross-validation**: Temporal splits (5 folds)
- **Feature scaling**: StandardScaler
- **Missing values**: Adaptive imputation

### SHAP Analysis
- **Explainer**: TreeExplainer for Random Forest
- **Sample size**: 1,000 samples (for performance)
- **Feature importance**: Top 10 features per biomarker

## 🚨 Error Handling

Common issues and solutions:

### Data Type Errors
- Mixed types in coordinates → Converted to numeric
- Date format inconsistencies → Pandas datetime parsing
- Missing value indicators → Proper NaN handling

### Memory Management
- Large dataset processing → Efficient pandas operations
- SHAP calculation limits → Sampling for explainability
- Model storage → Selective result saving

### Temporal Validation
- Data leakage prevention → TimeSeriesSplit cross-validation
- Temporal ordering → Sort by date/year before analysis
- Future data contamination → Strict temporal boundaries

## 📈 Performance Monitoring

The pipeline tracks:
- **Execution time** for each module
- **Memory usage** during processing
- **Model performance** across biomarkers
- **Data quality** metrics at each step

## 🔄 Pipeline Extensions

To add new features:

1. **New biomarkers**: Add to target_variables in preprocessor
2. **Climate features**: Extend climate engineering in preprocessor
3. **ML models**: Modify model initialization in analyzer
4. **XAI methods**: Add explainers in analyzer module

---

*This modular pipeline ensures reproducible, scalable climate-health analysis with publication-quality results.*