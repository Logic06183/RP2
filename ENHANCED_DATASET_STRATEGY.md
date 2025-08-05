# Enhanced Dataset Extraction Strategy for Heat-Health XAI Analysis

## Executive Summary

Analysis of raw data in `incoming/RP2` and climate resources in `selected_data_all/data/RP2_subsets` reveals **tremendous untapped potential** for creating superior datasets compared to the current processed `climate_linked` versions. The raw data contains **150+ biomarkers across comprehensive pathway panels** with **3,000+ participants** when properly integrated.

## 🔍 **Key Discoveries**

### **Raw Data Richness vs. Processed Versions**

| Dataset | Current (Processed) | Raw Data Potential | Enhancement Factor |
|---------|-------------------|-------------------|-------------------|
| **JHB_DPHRU_053** | 133 variables, 159 participants | 200+ variables, 1,013 participants | **6.4x participants** |
| **JHB_WRHI_001** | Limited pathway coverage | 25+ biomarkers × 11,320 lab records | **Comprehensive labs** |
| **ACTG Studies** | 87-76-20 participants | 200+ each with full panels | **10x larger cohorts** |
| **Climate Linkage** | Single daily averages | Hourly multi-source data | **24x temporal resolution** |

### **Lost Biomarker Treasure Trove**

#### **JHB_DPHRU_053 (MASC Study) Raw Variables:**
- **Metabolic Panel**: `glucose_0`, `total_cholesterol`, `triglycerides`, `hdl`, `ldl`, `hba1c`, `insulin`
- **Inflammatory**: `crp`, `inflammatory_markers`, `white_cell_differentials`
- **Cardiovascular**: `systolic_bp_average`, `diastolic_bp_average`, `pulse`, `ecg_parameters`
- **Renal Function**: `creatinine`, `bun`, `electrolytes`, `urinalysis_panel`
- **Body Composition**: DEXA scan data, fat mass, lean mass, bone density
- **Lifestyle**: Sleep quality, physical activity, dietary intake, stress measures

#### **JHB_WRHI_001 (D4T Trial) Raw Structure:**
```
ADLB.csv: 25+ chemistry biomarkers × multiple visits
├── CHEMISTRY-GLUCOSE, HDL, LDL, TRIGLYCERIDES
├── HEMATOLOGY-WBC, RBC, PLATELETS, NEUTROPHILS
├── LIVER FUNCTION-ALT, AST, BILIRUBIN
├── RENAL-CREATININE, BUN, ELECTROLYTES
└── IMMUNOLOGY-CD4, VIRAL_LOAD, CYTOKINES
```

### **Superior Climate Data Resources**

#### **RP2_subsets/JHB/ Contains:**
- **ERA5 Reanalysis**: Hourly temperature, humidity, pressure, wind
- **WRF Downscaled**: 3km resolution Johannesburg-specific modeling
- **MODIS LST**: Satellite land surface temperature (day/night)
- **SAAQIS Integration**: Air quality + meteorological variables
- **Multiple Heat Indices**: Pre-calculated WBGT, Heat Index, Humidex

## 🚀 **Enhanced Dataset Creation Plan**

### **Phase 1: Comprehensive Raw Data Extraction (Weeks 1-2)**

#### **Priority Dataset: JHB_DPHRU_053 Enhanced**
```yaml
enhanced_dphru053:
  source: "incoming/RP2/JHB_DPHRU_053/"
  target_size: "1,013 participants (vs 159 current)"
  biomarker_expansion: "200+ variables (vs 133 current)"
  
  pathways:
    metabolic:
      targets: ["glucose_0", "hba1c", "insulin", "total_cholesterol", "triglycerides", "hdl", "ldl"]
      completeness: "95%+ (vs 70% current)"
    
    inflammatory:
      targets: ["crp", "wbc", "neutrophils", "lymphocytes", "monocytes"]
      completeness: "90%+ (vs 60% current)"
    
    cardiovascular:
      targets: ["systolic_bp_average", "diastolic_bp_average", "pulse", "ecg_measures"]
      completeness: "95%+ (vs 80% current)"
    
    renal:
      targets: ["creatinine", "bun", "sodium", "potassium", "chloride"]
      completeness: "90%+ (vs 40% current)"
    
    body_composition:
      targets: ["dexa_fat_mass", "dexa_lean_mass", "bone_density", "android_fat", "gynoid_fat"]
      completeness: "85%+ (completely missing current)"
```

#### **Priority Dataset: JHB_WRHI_001 Enhanced**
```yaml
enhanced_wrhi001:
  source: "incoming/RP2/JHB_WRHI_001/"
  structure: "Multi-table clinical trial database"
  lab_records: "11,320 comprehensive lab results"
  
  biomarker_density:
    chemistry_panel: "25+ biomarkers per visit"
    hematology_panel: "15+ biomarkers per visit"
    immunology_panel: "CD4, viral loads, cytokines"
    liver_function: "ALT, AST, bilirubin, albumin"
    
  temporal_structure:
    visits_per_participant: "4-8 visits"
    follow_up_duration: "48+ weeks"
    visit_level_climate_linkage: "Precise exposure timing"
```

### **Phase 2: Multi-Source Climate Integration (Weeks 2-3)**

#### **Enhanced Climate Linkage Strategy**
```python
climate_integration_pipeline = {
    'sources': {
        'ERA5': {
            'path': 'selected_data_all/data/RP2_subsets/JHB/ERA5_*.zarr',
            'variables': ['temperature', 'humidity', 'pressure', 'wind'],
            'resolution': 'hourly',
            'coverage': '1979-present'
        },
        'WRF': {
            'path': 'selected_data_all/data/RP2_subsets/JHB/WRF_*.zarr',
            'variables': ['temperature', 'lst'],
            'resolution': '3km spatial, hourly',
            'coverage': 'Johannesburg metro'
        },
        'MODIS': {
            'path': 'selected_data_all/data/RP2_subsets/JHB/modis_lst_*.zarr',
            'variables': ['day_lst', 'night_lst'],
            'resolution': '1km spatial, daily',
            'coverage': 'satellite-derived'
        },
        'SAAQIS': {
            'path': 'selected_data_all/data/RP2_subsets/JHB/SAAQIS_*.zarr',
            'variables': ['air_quality', 'meteorology'],
            'resolution': 'station-based, hourly',
            'coverage': 'Johannesburg monitoring network'
        }
    },
    
    'processing': {
        'temporal_matching': 'visit-date precise linkage',
        'lag_windows': [1, 3, 7, 14, 21, 28, 30, 60, 90],
        'aggregations': ['mean', 'max', 'min', 'range', 'cumulative'],
        'heat_indices': ['wbgt', 'heat_index', 'humidex', 'utci', 'apparent_temp']
    }
}
```

### **Phase 3: ACTG Multi-Study Integration (Weeks 3-4)**

#### **Standardized ACTG Panel Integration**
```yaml
actg_integration:
  studies: ["JHB_ACTG_015", "JHB_ACTG_016", "JHB_ACTG_017"]
  combined_sample: "300+ participants (vs 183 current)"
  
  standardized_biomarkers:
    immunological: ["cd4_count", "cd4_percent", "viral_load", "hiv_rna"]
    metabolic: ["glucose", "lipid_panel", "liver_enzymes"]
    renal: ["creatinine", "bun", "electrolytes"]
    hematological: ["complete_blood_count", "differential"]
    
  harmonization_approach:
    units: "Standardize across studies"
    visit_timing: "Map to standardized schedule"
    reference_ranges: "Use consistent normal values"
```

### **Phase 4: Pathway-Specific XAI Optimization (Weeks 4-5)**

#### **Enhanced Pathway Target Creation**
```python
enhanced_pathways = {
    'inflammatory_comprehensive': {
        'target_creation': 'multi_biomarker_composite',
        'biomarkers': ['crp', 'wbc', 'neutrophils', 'lymphocytes', 'monocytes', 'cytokines'],
        'scoring': 'weighted_z_score_composite',
        'validation': 'clinical_threshold_mapping'
    },
    
    'metabolic_syndrome': {
        'target_creation': 'metabolic_syndrome_score',
        'biomarkers': ['glucose', 'hba1c', 'insulin', 'triglycerides', 'hdl', 'waist_circumference'],
        'scoring': 'aha_nhlbi_criteria_based',
        'validation': 'diabetes_hypertension_outcomes'
    },
    
    'cardiovascular_risk': {
        'target_creation': 'cv_risk_composite',
        'biomarkers': ['systolic_bp', 'diastolic_bp', 'pulse_pressure', 'total_cholesterol', 'ldl'],
        'scoring': 'framingham_risk_adapted',
        'validation': 'hypertension_diagnosis'
    },
    
    'renal_function': {
        'target_creation': 'egfr_comprehensive',
        'biomarkers': ['creatinine', 'bun', 'electrolytes', 'proteinuria'],
        'scoring': 'ckd_epi_equation',
        'validation': 'kidney_disease_staging'
    }
}
```

## 📊 **Expected Outcomes from Enhanced Strategy**

### **Dataset Quality Improvements**
| Metric | Current | Enhanced | Improvement |
|--------|---------|----------|-------------|
| **Total Participants** | ~1,500 | ~4,000+ | **2.7x increase** |
| **Biomarker Variables** | ~50 | ~200+ | **4x expansion** |
| **Pathway Completeness** | 40-60% | 90-95% | **Complete pathways** |
| **Temporal Resolution** | Daily averages | Hourly/visit-level | **24x precision** |
| **Climate Sources** | ERA5 only | 4 integrated sources | **Comprehensive coverage** |
| **Multi-study Integration** | Single studies | Harmonized cohorts | **Cross-validation** |

### **XAI Analysis Enhancement**
1. **Richer Feature Space**: 200+ biomarkers enable comprehensive pathway modeling
2. **Temporal Dynamics**: Visit-level data captures biomarker evolution
3. **Multi-pathway Interactions**: Complete pathway panels enable cross-system analysis
4. **Climate Precision**: Hour-level exposure data improves mechanism discovery
5. **Larger Sample Size**: 4,000+ participants provide robust statistical power

### **Scientific Impact Potential**
- **Novel Biomarker Discovery**: Body composition, sleep, dietary factors
- **Temporal Mechanism Elucidation**: How heat exposure evolves into health impacts
- **Multi-system Integration**: Cross-pathway heat vulnerability patterns
- **Population Heterogeneity**: HIV+ vs general population responses
- **Climate Source Validation**: Compare satellite vs reanalysis vs station data

## 🔧 **Implementation Pipeline**

### **Technical Infrastructure**
```python
# Enhanced data extraction pipeline
class EnhancedDataExtractor:
    def __init__(self):
        self.raw_data_paths = {
            'dphru053': 'incoming/RP2/JHB_DPHRU_053/',
            'wrhi001': 'incoming/RP2/JHB_WRHI_001/',
            'actg_studies': 'incoming/RP2/JHB_ACTG_*/',
            'climate_sources': 'selected_data_all/data/RP2_subsets/JHB/'
        }
    
    def extract_comprehensive_biomarkers(self, study):
        """Extract all available biomarkers maintaining pathway structure."""
        pass
    
    def integrate_multi_source_climate(self, temporal_points):
        """Combine ERA5, WRF, MODIS, SAAQIS data."""
        pass
    
    def create_visit_level_linkage(self, health_data, climate_data):
        """Precise temporal matching for exposure assessment."""
        pass
    
    def harmonize_across_studies(self, study_datasets):
        """Standardize biomarkers and units across ACTG studies."""
        pass
```

### **Quality Assurance Framework**
1. **Biomarker Validation**: Check against clinical reference ranges
2. **Temporal Consistency**: Ensure proper visit sequencing
3. **Climate Data Quality**: Validate against known weather patterns
4. **Cross-study Harmonization**: Verify consistent variable definitions
5. **Missing Data Assessment**: Document and handle missingness patterns

## 🎯 **Implementation Priority**

### **Week 1-2: Foundation**
1. **Extract JHB_DPHRU_053 comprehensive data** (1,013 participants, 200+ variables)
2. **Integrate multi-source climate data** for Johannesburg region
3. **Create visit-level temporal linkage** system

### **Week 3-4: Expansion**
1. **Add JHB_WRHI_001 lab-rich dataset** (11,320 lab records)
2. **Harmonize ACTG studies** (015, 016, 017)
3. **Build comprehensive pathway targets**

### **Week 5-6: Integration**
1. **Multi-study dataset integration**
2. **Enhanced XAI pipeline deployment**
3. **Cross-validation across cohorts**

### **Week 7-8: Optimization**
1. **Pathway-specific model optimization**
2. **Novel biomarker integration** (body composition, lifestyle)
3. **Comprehensive results validation**

## 🏆 **Success Metrics**

### **Quantitative Targets**
- **Sample Size**: >3,000 participants (vs <1,500 current)
- **Biomarker Coverage**: >90% pathway completeness (vs 40-60% current)
- **Temporal Resolution**: Visit-level climate matching (vs enrollment averages)
- **Multi-pathway R²**: >0.20 (vs 0.149 current best)

### **Qualitative Improvements**
- **Novel Hypothesis Generation**: Body composition × climate interactions
- **Temporal Mechanism Discovery**: Heat exposure → biomarker evolution
- **Multi-system Integration**: Cross-pathway vulnerability patterns
- **Population Insights**: HIV+ heat vulnerability characteristics

This enhanced strategy transforms the heat-health XAI analysis from a limited single-dataset approach to a comprehensive multi-cohort, multi-biomarker, multi-climate-source investigation capable of revealing novel heat-health mechanisms and generating actionable public health insights.

---

*Enhanced dataset strategy leveraging full raw data potential for comprehensive heat-health XAI analysis.*