# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Final Validated Submission** for the climate-health analysis of Johannesburg, South Africa. This folder contains the validated manuscript, publication-ready figures, and supplementary materials analyzing health impacts of heat exposure using machine learning across 9,103 individuals from 17 research cohorts (2002-2021).

## Key Commands

### Installation and Setup
```bash
# Navigate to parent directory for full codebase access
cd ../

# Install dependencies (if needed)
pip install -r requirements.txt

# Verify Python analysis tools
python -c "import pandas, numpy, matplotlib, scipy; print('✅ Core packages ready')"
```

### Running Analysis Scripts
```bash
# Generate publication figures from parent directory
cd ../
python create_publication_graphics.py

# Run detailed effect analysis
python detailed_effect_analysis.py

# Run socioeconomic vulnerability analysis
python socioeconomic_vulnerability_analysis.py

# Run comprehensive heat-health analysis
python comprehensive_heat_health_analysis.py
```

### Data Analysis Commands
```bash
# Run biomarker analysis
python biomarker_analysis.py

# Run climate data analysis
python climate_data_analysis.py

# Run comprehensive data analysis
python comprehensive_data_analysis.py

# Run socioeconomic vulnerability analysis (uses GCRO data)
python socioeconomic_vulnerability_analysis.py
```

### Advanced ML Analysis
```bash
# Run the advanced ML climate health analysis
python advanced_ml_climate_health_analysis.py

# Run publication-ready analysis
python publication_ready_analysis.py

# Run final corrected analysis
python final_corrected_analysis.py
```

## Architecture and Key Components

### Current Directory Structure (`FINAL_VALIDATED_SUBMISSION/`)
- **`manuscript/`**: Contains `RIGOROUS_VALIDATED_MANUSCRIPT.md` - the final validated manuscript
- **`figures/`**: Publication-ready SVG figures
  - `main_findings_comprehensive.svg`: Primary results visualization
  - `before_after_comparison.svg`: Methodological transformation
  - `clinical_significance_assessment.svg`: Clinical relevance analysis
  - `socioeconomic_vulnerability.svg`: Environmental justice findings
  - `comprehensive_study_overview.svg`: Study overview infographic
- **`supplementary/`**: Contains `EXPERT_VALIDATED_ANALYSIS.md` - detailed methodology

### Parent Directory Analysis Scripts (`../`)
Key analysis scripts that generated the results:
- **`create_publication_graphics.py`**: Generates all publication figures
- **`comprehensive_heat_health_analysis.py`**: Main heat-health analysis pipeline
- **`socioeconomic_vulnerability_analysis.py`**: SE vulnerability analysis
- **`detailed_effect_analysis.py`**: Detailed biomarker effect analysis
- **`advanced_ml_climate_health_analysis.py`**: Advanced ML methods implementation

### Data Processing Pipeline (`../data/`)
- **`climate_data_analysis.py`**: Climate variable processing
- **`biomarker_analysis.py`**: Biomarker data extraction and analysis
- **`comprehensive_data_analysis.py`**: Integrated data analysis

### Real Data Sources
- **Health Data**: `/home/cparker/selected_data_all/data/` - RP2 harmonized health datasets
- **Climate Data**: `/home/cparker/selected_data_all/data/RP2_subsets/JHB/` - ERA5, WRF, MODIS, SAAQIS climate data (zarr format)
- **Socioeconomic Data**: `/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv` - GCRO Quality of Life Survey with climate integration

### Machine Learning Components (`../`)
- **`final_corrected_analysis.py`**: Final validated ML analysis
- **`publication_ready_analysis.py`**: Publication-ready results generation
- **`enhanced_analysis_final.py`**: Enhanced analysis with all corrections

## Key Analysis Parameters

### Model Configuration
- **ML Methods**: RandomForest, GradientBoosting, Ridge, ElasticNet
- **Cross-Validation**: Temporal cross-validation to prevent leakage
- **Sample Size**: 9,103 individuals (correctly calculated, not inflated)
- **Effect Sizes**: Conservative (R² = 0.015-0.034)

### Climate Variables (from RP2_subsets/JHB/)
- **ERA5 reanalysis data**: Temperature, wind speed (native and regridded zarr files)
- **WRF downscaled data**: High-resolution temperature (3km resolution)
- **MODIS satellite data**: Land surface temperature observations
- **SAAQIS stations**: Local weather station data with climate variables
- **Derived metrics**: Heat indices (WBGT, Heat Index), lag periods (7, 14, 21, 28 days), rolling windows (3, 7, 14 days)

### Biomarker Targets
- **Glucose**: Primary metabolic marker
- **CRP**: Inflammatory response
- **Blood Pressure**: Cardiovascular markers
- **Creatinine**: Renal function

## Important Patterns and Conventions

### Real Data Usage (VERIFIED ✅)
- **Climate Data**: Uses actual ERA5 reanalysis from zarr files in `/home/cparker/selected_data_all/data/RP2_subsets/JHB/`
- **Health Data**: Uses real RP2 harmonized clinical datasets from `/home/cparker/selected_data_all/data/`
- **Socioeconomic Data**: Uses actual GCRO Quality of Life Survey with pre-integrated ERA5 climate variables
- **NO SIMULATION**: All analysis scripts corrected to remove artificial data generation
- **Key Scripts Verified**: `advanced_ml_climate_health_analysis.py`, `comprehensive_heat_health_analysis.py`, `socioeconomic_vulnerability_analysis.py`

### Statistical Reporting
- Always report conservative effect sizes
- Use proper sample size calculations (unique individuals, not observations)
- Apply clinical significance thresholds
- Report confidence intervals and p-values

### Figure Generation
- All figures in SVG format for publication quality
- Consistent color scheme across visualizations
- Statistical significance clearly marked
- Effect sizes with confidence intervals

### Manuscript Standards
- Rigorous methodology section
- Complete transparency on limitations
- Expert validation incorporated
- No exaggeration of findings

## Common Development Tasks

### Update Figures
```bash
cd ../
python create_publication_graphics.py
# Figures saved to FINAL_VALIDATED_SUBMISSION/figures/
```

### Regenerate Analysis
```bash
cd ../
python final_corrected_analysis.py
python publication_ready_analysis.py
```

### Check Data Quality
```bash
cd ../
python comprehensive_data_analysis.py
# Review comprehensive_data_quality_report.json

# Check actual data sources
ls -la /home/cparker/selected_data_all/data/RP2_subsets/JHB/  # Climate zarr files
ls -la /home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/  # GCRO socioeconomic data
```

### Validate Results
```bash
cd ../
python biomarker_analysis.py
# Check biomarker_availability_report.json for data coverage

# Verify real data usage (CORRECTED)
python advanced_ml_climate_health_analysis.py  # Now uses real GCRO+ERA5 data
python comprehensive_heat_health_analysis.py   # Uses real GCRO data
python socioeconomic_vulnerability_analysis.py # Uses real GCRO data
```

### Verify Real Data Sources
```bash
# Confirm climate data exists
ls -la /home/cparker/selected_data_all/data/RP2_subsets/JHB/*.zarr

# Confirm GCRO data exists and has climate integration
python3 -c "import pandas as pd; df = pd.read_csv('/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv'); print(f'GCRO: {len(df)} records, {len([c for c in df.columns if \"era5\" in c])} climate variables')"
```

## GitHub Integration

### Prepare for Upload
```bash
# Ensure all files are in correct structure
ls -la manuscript/
ls -la figures/
ls -la supplementary/
```

### Commit Changes
```bash
git add FINAL_VALIDATED_SUBMISSION/
git commit -m "Update validated manuscript and figures"
git push origin main
```

## Key Files to Edit

### Manuscript Updates
- Main manuscript: `manuscript/RIGOROUS_VALIDATED_MANUSCRIPT.md`
- Supplementary: `supplementary/EXPERT_VALIDATED_ANALYSIS.md`

### Figure Regeneration
- Edit and run: `../create_publication_graphics.py`
- Output location: `figures/`

### Analysis Updates
- Core analysis: `../final_corrected_analysis.py`
- Publication prep: `../publication_ready_analysis.py`

## Performance Considerations

- Large dataset processing uses efficient pandas operations
- SVG figures optimized for file size
- Analysis scripts use vectorized operations where possible
- Conservative memory usage for compatibility

## Data Integration Architecture

### Primary Data Sources
1. **Health cohorts**: RP2 harmonized datasets from `/home/cparker/selected_data_all/data/`
   - 17 clinical studies with health biomarkers
   - 9,103 individuals across Johannesburg area
   - Standardized to HEAT Master Codebook (116 variables)

2. **Climate datasets**: Multi-source integration from `/home/cparker/selected_data_all/data/RP2_subsets/JHB/`
   - **ERA5-Land**: Native land surface temperature (zarr format)
   - **ERA5**: Temperature and wind speed data (native + regridded)
   - **WRF**: High-resolution downscaled climate (3km)
   - **MODIS**: Satellite land surface temperature
   - **SAAQIS**: Weather station network with climate variables

3. **Socioeconomic data**: GCRO Quality of Life Survey
   - Path: `/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv`
   - 500 responses with integrated 30-day climate exposure windows
   - Income, education, employment, healthcare access variables

## Quality Assurance

### Before Submission
1. Verify all figures are up-to-date
2. Check manuscript for consistency with figures
3. Ensure supplementary materials are complete
4. Validate all statistical claims
5. Review expert critique responses

### Key Validation Points
- Sample size: 9,103 unique individuals (not inflated)
- Effect sizes: R² = 0.015-0.034 (conservative)
- Statistical methods: Appropriate for data structure
- Clinical significance: Properly assessed
- Limitations: Fully documented