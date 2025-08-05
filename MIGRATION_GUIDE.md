# Migration Guide: From Original Repository to Optimized Framework

This guide helps you migrate from the original `Min_repo_heat_analysis` to the optimized `heat_analysis_optimized` framework.

## 🔄 What Changed

### Repository Structure
```
OLD: Min_repo_heat_analysis/
├── 70+ Python files (many redundant)
├── heat_vulnerability/ (basic module)
├── outputs/ (mixed results)
└── data/ (embedded in repo)

NEW: heat_analysis_optimized/
├── core/ (pipeline + config)
├── data/ (loading + validation)
├── features/ (engineering modules)
├── targets/ (pathway targets)
├── models/ (optimization)
├── explainability/ (XAI analysis)
├── datasets/ (configurations)
└── example_analysis.py
```

### Key Improvements

1. **Reduced Codebase**: From 70+ files to ~15 core modules
2. **Multi-Dataset Support**: YAML-based dataset configurations
3. **Modular Architecture**: Clean separation of concerns
4. **Scalable Pipeline**: Easy to add new datasets and pathways
5. **Standardized XAI**: Consistent SHAP analysis across datasets
6. **Hypothesis Generation**: Automated scientific hypothesis creation

## 📊 File Mapping

### Core Analysis Files
| Original | New Location | Status |
|----------|-------------|--------|
| `improved_heat_health_xai_pipeline.py` | `core/pipeline.py` | ✅ Refactored |
| `climate_health_interaction_analysis.py` | `core/pipeline.py` | ✅ Integrated |
| `config.py` | `core/config.py` | ✅ Enhanced |

### Feature Engineering
| Original | New Location | Status |
|----------|-------------|--------|
| `heat_vulnerability/feature_engineering.py` | `features/climate.py`, `features/temporal.py`, `features/interactions.py` | ✅ Modularized |

### XAI Analysis
| Original | New Location | Status |
|----------|-------------|--------|
| `final_shap_plots.py` | `explainability/shap_analysis.py` | ✅ Refactored |
| `leak_free_shap_plots.py` | `explainability/shap_analysis.py` | ✅ Integrated |
| Various SHAP files | `explainability/hypothesis.py` | ✅ Consolidated |

### Data Processing
| Original | New Location | Status |
|----------|-------------|--------|
| `heat_vulnerability/data_loader.py` | `data/loader.py` | ✅ Enhanced |
| Manual data paths | `datasets/*.yaml` | ✅ Configurable |

### Removed Files (37 files)
- `basic_shap_plots.py`, `simple_shap_plots.py` → Superseded by unified XAI
- `create_*_target.py` files → Replaced by `targets/pathways.py`
- `run_*_pipeline.py` files → Replaced by single `core/pipeline.py`
- Diagnostic files → Not needed in production framework

## 🚀 Migration Steps

### 1. Update Analysis Code
```python
# OLD APPROACH
from improved_heat_health_xai_pipeline import improved_heat_health_xai_analysis
results = improved_heat_health_xai_analysis()

# NEW APPROACH
from core.pipeline import HeatHealthAnalyzer
analyzer = HeatHealthAnalyzer(dataset_name='dphru053')
results = analyzer.run_analysis()
```

### 2. Dataset Configuration
```python
# OLD: Hardcoded paths in scripts
data_path = '/Users/craig/Desktop/ML_paper/Min_repo_heat_analysis/data/...'

# NEW: YAML configuration
# datasets/dphru053.yaml defines all paths and settings
analyzer = HeatHealthAnalyzer(dataset_name='dphru053')
```

### 3. Multi-Dataset Analysis
```python
# OLD: Manual script copying and editing
# NEW: Seamless multi-dataset comparison
for dataset in ['dphru053', 'vida007', 'actg015']:
    analyzer = HeatHealthAnalyzer(dataset_name=dataset)
    results[dataset] = analyzer.run_analysis()
```

### 4. Custom Configuration
```python
# OLD: Edit config.py file directly
# NEW: Runtime configuration override
custom_config = {
    'lag_periods': [7, 14, 21],
    'shap_sample_size': 500
}
analyzer = HeatHealthAnalyzer(
    dataset_name='dphru053',
    custom_config=custom_config
)
```

## 📋 Migration Checklist

### Data Migration
- [ ] Copy dataset files to new `data/` directory
- [ ] Create YAML configuration file using `datasets/template.yaml`
- [ ] Test data loading with `DataLoader`
- [ ] Validate data quality scores

### Analysis Migration
- [ ] Replace hardcoded analysis scripts with `HeatHealthAnalyzer`
- [ ] Update pathway definitions in dataset config
- [ ] Test feature engineering pipeline
- [ ] Verify model optimization results

### Results Migration
- [ ] Update output directory structure
- [ ] Migrate existing analysis results to new format
- [ ] Test report generation functionality
- [ ] Validate XAI analysis consistency

### New Features Adoption
- [ ] Configure hypothesis generation parameters
- [ ] Set up multi-dataset comparison workflows
- [ ] Implement automated quality validation
- [ ] Establish standardized reporting pipeline

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Add to Python path if needed
import sys
sys.path.append('/path/to/heat_analysis_optimized')
```

**2. Missing Dependencies**
```bash
pip install -r requirements.txt
```

**3. Data Path Issues**
```yaml
# In dataset YAML, use absolute paths
dataset:
  file_path: "/absolute/path/to/your/dataset.csv"
```

**4. Feature Name Mismatches**
- Check column prefixes in dataset configuration
- Validate feature engineering results
- Review pathway target availability

### Performance Considerations

**Memory Usage**
- New framework is more memory efficient
- Configurable sample sizes for SHAP analysis
- Streaming data processing for large datasets

**Processing Speed**
- Parallel model optimization
- Reduced redundant computations
- Configurable analysis depth

## 📈 Benefits of Migration

### For Single Dataset Analysis
- ✅ Cleaner, more maintainable code
- ✅ Standardized output formats
- ✅ Enhanced XAI capabilities
- ✅ Automated hypothesis generation

### For Multi-Dataset Analysis
- ✅ Seamless dataset integration
- ✅ Consistent methodology across cohorts
- ✅ Automated cross-dataset comparison
- ✅ Scalable to 10+ datasets

### For Research Team
- ✅ Reduced code duplication
- ✅ Easier onboarding for new researchers
- ✅ Standardized scientific reporting
- ✅ Version-controlled methodology

## 🎯 Next Steps

1. **Start with Single Dataset**: Migrate your primary analysis using DPHRU053
2. **Validate Results**: Compare outputs with original analysis for consistency
3. **Add New Datasets**: Create configurations for additional cohorts
4. **Explore New Features**: Use hypothesis generation and enhanced XAI
5. **Scale Analysis**: Run comparative studies across multiple populations

## 📞 Support

For migration support:
- Review `example_analysis.py` for usage patterns
- Check dataset configurations in `datasets/`
- Examine module documentation in each Python file
- Test with small datasets first

The optimized framework maintains all the scientific rigor of the original analysis while providing a scalable foundation for multi-dataset heat-health research.