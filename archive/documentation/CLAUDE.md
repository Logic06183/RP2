# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Heat-Health XAI Analysis Framework** - a comprehensive explainable AI system for analyzing heat-health-socioeconomic interactions in African urban populations. The project analyzes health impacts of heat exposure using machine learning and SHAP-based explainability across multiple cohorts in Johannesburg, South Africa.

## Key Commands

### Installation and Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "from core.pipeline import HeatHealthAnalyzer; print('✅ Framework ready')"
```

### Running Analysis
```bash
# Run example analysis on DPHRU053 dataset
python example_analysis.py

# Run specific pathway analysis
python -c "from core.pipeline import HeatHealthAnalyzer; analyzer = HeatHealthAnalyzer('dphru053'); analyzer.run_analysis(['inflammatory'])"

# Generate publication figures
python analysis/create_publication_figures.py

# Run state-of-art XAI analysis
python analysis/state_of_art_xai_analyzer.py
```

### Data Processing
```bash
# Run complete data extraction pipeline
python data/run_complete_extraction.py

# Create XAI-ready datasets
python data/optimal_xai_dataset_creator.py

# Integrate socioeconomic data
python data/socioeconomic_integrator.py
```

### Testing and Validation
```bash
# Run tests (when available)
pytest tests/

# Validate data quality
python -c "from heat_xai_framework.datasets import validate_dataset; validate_dataset('dphru053.yaml')"

# Check code formatting
black . --check
flake8 .
```

## Architecture and Key Components

### Core Pipeline (`core/`)
- **`pipeline.py`**: Main `HeatHealthAnalyzer` class orchestrating the entire analysis workflow
- **`config.py`**: Configuration management for datasets and analysis parameters
- Supports multiple datasets through YAML configurations in `datasets/`

### Data Processing (`data/`)
- **Multi-stage extraction pipeline**: Raw → Enhanced → Integrated → XAI-ready datasets
- **Key extractors**:
  - `dataset_specific_extractors.py`: Core dataset extraction logic
  - `climate_integration.py`: Climate data linkage
  - `socioeconomic_integrator.py`: SE variable integration
  - `optimal_xai_dataset_creator.py`: Final XAI-ready dataset creation
- **Output hierarchy**: `comprehensive_extracted/` → `enhanced_**/` → `optimal_xai_ready/`

### Feature Engineering (`features/`)
- **`climate.py`**: Heat indices (WBGT, Heat Index), temperature features
- **`temporal.py`**: Lag features (7, 14, 21, 28 days), rolling windows
- **`interactions.py`**: Climate×BMI, Climate×Age, other cross-domain interactions

### Target Creation (`targets/`)
- **`pathways.py`**: Pathway-specific targets with leak prevention
  - Inflammatory (CRP-based)
  - Metabolic (glucose regulation)
  - Cardiovascular (blood pressure)
  - Renal (kidney function)

### Machine Learning (`models/`)
- **`optimization.py`**: Hyperparameter tuning with Optuna
- Supports RandomForest, GradientBoosting, ElasticNet
- Temporal cross-validation to prevent data leakage

### Explainability (`explainability/`)
- **`shap_analysis.py`**: SHAP value computation and interpretation
- **`hypothesis.py`**: Novel hypothesis generation from XAI insights
- Generates feature importance, interactions, and temporal patterns

### Analysis Notebooks
- **`heat_health_xai_showcase.ipynb`**: Main demonstration notebook
- **`heat_health_dlnm_analysis.ipynb`**: DLNM (Distributed Lag Non-linear Models) analysis
- **`dlnm_validation_simple.ipynb`**: Validation of DLNM results
- Multiple working notebooks for development and testing

## Dataset Structure

### Available Datasets
- **DPHRU053**: Primary dataset with comprehensive health markers
- **VIDA007/008**: Additional cohorts
- **WRHI001**: Biomarker-rich dataset
- **ACTG015-019**: Clinical trial datasets

### Dataset Configuration (`datasets/*.yaml`)
Each dataset has a YAML config specifying:
- File paths and column mappings
- Pathway targets and transformations
- Feature engineering parameters
- Analysis settings

### Data Flow
1. Raw data from `/terra/projects/heat_center/` (if available)
2. Extracted to `data/comprehensive_extracted/`
3. Enhanced with climate data to `data/enhanced_**/`
4. Integrated with SE variables to `data/socioeconomic_integrated/`
5. Final XAI-ready datasets in `data/optimal_xai_ready/`

## Important Patterns and Conventions

### Pathway-Specific Analysis
- Always exclude related biomarkers from predictors to prevent target leakage
- Each pathway has specific exclusion lists defined in dataset configs
- R² > 0.01 threshold for meaningful XAI analysis

### Temporal Features
- Standard lag periods: 7, 14, 21, 28 days
- Rolling windows: 3, 7, 14 days
- 21-day window shown to be optimal for heat adaptation

### Climate Features
- Temperature, humidity, heat indices (WBGT, Heat Index)
- Diurnal temperature range
- Temperature variability metrics
- All climate data integrated at daily level

### XAI Analysis
- SHAP analysis only on models with R² > 0.01
- Sample size of 1000 for SHAP computation
- Hypothesis generation for novel insights
- Feature interactions discovery

## Key Files to Edit

### Adding New Datasets
1. Create config: `datasets/new_dataset.yaml` (use `template.yaml`)
2. Add extractor if needed: `data/dataset_specific_extractors.py`
3. Update pipeline: `data/run_complete_extraction.py`

### Modifying Analysis
- Main pipeline: `core/pipeline.py`
- Feature engineering: `features/climate.py`, `features/temporal.py`
- Model optimization: `models/optimization.py`
- XAI analysis: `explainability/shap_analysis.py`

### Creating Reports
- Analysis reports: `analysis/` directory
- Publication figures: `analysis/create_publication_figures.py`
- Results visualization: notebooks in root directory

## Common Development Tasks

### Run Quick Analysis
```python
from core.pipeline import HeatHealthAnalyzer

analyzer = HeatHealthAnalyzer('dphru053')
results = analyzer.run_analysis(['inflammatory'], explain_predictions=False)
```

### Process New Dataset
```python
# 1. Extract raw data
from data.dataset_specific_extractors import extract_dataset
df = extract_dataset('new_dataset')

# 2. Integrate climate
from data.climate_integration import integrate_climate
df_climate = integrate_climate(df)

# 3. Create XAI-ready version
from data.optimal_xai_dataset_creator import create_xai_ready
df_xai = create_xai_ready(df_climate)
```

### Generate Custom Report
```python
from core.pipeline import HeatHealthAnalyzer

analyzer = HeatHealthAnalyzer('dphru053')
results = analyzer.run_analysis(pathways=['all'])
analyzer.generate_report(results, output_dir='custom_reports/')
```

## Performance Considerations

- Large datasets: Use sampling for initial exploration
- SHAP computation: Expensive for large datasets, use `shap_sample_size` parameter
- Parallel processing: Models use n_jobs=-1 for parallel computation
- Memory management: Data pipeline processes in chunks when needed

## Output Structure

### Analysis Results (`analysis/xai_results/`)
- Model performance metrics
- SHAP values and importance scores
- Generated hypotheses
- Visualization figures

### Processed Data (`data/optimal_xai_ready/`)
- `xai_ready_*.csv`: Analysis-ready datasets
- `XAI_DATASETS_SUMMARY.md`: Dataset statistics and quality metrics

### Reports (`analysis/`)
- Scientific papers and supplementary materials
- Policy insights and recommendations
- Publication-ready figures