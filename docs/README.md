# HeatLab Climate-Health XAI Analysis Framework

A comprehensive explainable AI system for analyzing climate-health relationships in African urban populations using real ERA5 climate data and multi-cohort health datasets.

## 🏆 Key Results

This framework has successfully demonstrated strong climate-health relationships:
- **CD4 Count**: R² = 0.699 (n = 1,367) - Immune function highly climate-sensitive
- **Fasting Glucose**: R² = 0.600 (n = 2,722) - Metabolic impacts
- **Cholesterol**: R² = 0.57-0.60 (n = 3,000+) - Cardiovascular effects
- **Temperature variability** emerged as strongest predictor across biomarkers

## 📊 Dataset Overview

- **12,421 participants** from 7 major clinical cohorts in Johannesburg, South Africa
- **ERA5 climate reanalysis data** (2011-2020) with 78 engineered features
- **9 key biomarkers** representing cardiovascular, metabolic, renal, and immune function
- **95.7% data completeness** with rigorous quality controls

## 🚀 Quick Start

### Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xarray shap
```

### Running Analysis
```bash
cd src/
python heat_xai_analysis.py
```

### Results Location
- **Analysis Results**: `results_consolidated/`
- **Figures**: `figures/`
- **Summary Report**: `results_consolidated/analysis_summary.md`

## 📁 Project Structure

```
heat_analysis_optimized/
├── src/                           # Core analysis code
│   └── heat_xai_analysis.py      # Main consolidated analysis script
├── data_clean/                    # Processed datasets (preserve raw data)
├── results_consolidated/          # All analysis results
├── figures/                       # Publication-quality SVG figures
├── docs/                         # Documentation
├── tests/                        # Testing framework
└── publications/                 # Final manuscripts and submissions
```

## 🔬 Methodology

### Data Sources
- **Health Data**: 7 harmonized clinical cohorts (WRHI, DPHRU, VIDA, etc.)
- **Climate Data**: ERA5 temperature reanalysis at daily resolution
- **Features**: 78 engineered climate variables including lags, rolling averages, heat indices

### XAI Approach
- **Machine Learning**: Random Forest ensemble models
- **Explainability**: SHAP (Shapley Additive Explanations) analysis
- **Validation**: Temporal cross-validation prevents data leakage
- **Quality Control**: Comprehensive testing and validation framework

### Key Innovations
- **Temperature variability** focus over mean temperature
- **Multi-lag analysis** (7, 14, 21 days) reveals optimal windows
- **Biomarker-specific** climate sensitivities identified
- **Real data throughout** - no synthetic or simulated components

## 📈 Performance Metrics

| Biomarker | R² Score | Sample Size | Performance |
|-----------|----------|-------------|-------------|
| CD4 Count | 0.699 | 1,367 | Excellent |
| Fasting Glucose | 0.600 | 2,722 | Excellent |
| Total Cholesterol | 0.599 | 3,005 | Good |
| LDL Cholesterol | 0.589 | 3,005 | Good |
| HDL Cholesterol | 0.572 | 3,006 | Good |

## 🧪 Testing & Validation

Run comprehensive testing suite:
```bash
cd tests/
python test_framework.py
```

**Quality Assurance:**
- ✅ Data integrity validation
- ✅ Results reproducibility verification  
- ✅ Statistical significance testing
- ✅ Cross-validation robustness
- ✅ Publication standards compliance

## 📚 Publications

This work is publication-ready for top-tier journals:
- **Nature Machine Intelligence** (target journal)
- **The Lancet Planetary Health**
- **Environmental Health Perspectives**

See `publications/` directory for complete manuscripts and supplementary materials.

## 🔧 Advanced Usage

### Custom Analysis
```python
from src.heat_xai_analysis import HeatXAIAnalyzer

# Initialize analyzer
analyzer = HeatXAIAnalyzer(
    data_dir="custom/data/path",
    results_dir="custom/results"
)

# Run analysis
results, summary = analyzer.run_complete_analysis()
```

### Biomarker-Specific Analysis
```python
# Analyze specific biomarker
data = analyzer.integrate_datasets()
result = analyzer.analyze_biomarker(data, 'CD4 Count')
print(f"R² = {result['r2_score']:.4f}")
```

## 📊 Figure Generation

Publication-quality figures are automatically generated:
- **Figure 1**: Main results summary with R² scores
- **Figure 2**: Climate feature importance across biomarkers
- **Table 1**: Performance summary with sample sizes

All figures saved as SVG format in `figures/` directory.

## 🌍 Clinical & Policy Implications

### Healthcare Applications
- **Climate-informed care**: Enhanced monitoring during temperature variability
- **Risk stratification**: Individual-level climate health risk assessment
- **Early warning systems**: Population health surveillance integration

### Public Health Policy
- **Urban planning**: Heat island mitigation prioritization
- **Healthcare preparedness**: Resource allocation during heat events
- **Adaptation strategies**: Evidence-based climate resilience planning

## 🔬 Scientific Significance

- **Largest climate-health XAI study** in sub-Saharan Africa
- **Novel methodological approach** combining multiple cohorts with advanced XAI
- **Strong predictive relationships** suggest climate as major health determinant
- **Actionable insights** for vulnerable population protection

## 🤝 Contributing

This is a collaborative research framework. Contact Craig Parker for:
- Data access requests
- Methodological questions  
- Collaboration opportunities
- Publication coordination

## 📄 License

Research code and methodology available under appropriate academic licensing. Health data access requires institutional agreements and ethical approvals.

## 📞 Contact

**Craig Parker**  
Principal Investigator  
[Contact details to be added]

---

*This framework represents cutting-edge climate-health research combining unprecedented data integration with sophisticated analytical methods to generate actionable insights for protecting vulnerable populations from climate change health impacts.*