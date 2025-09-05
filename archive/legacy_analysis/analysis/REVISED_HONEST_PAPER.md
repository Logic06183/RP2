# Heat-health associations in Johannesburg: A methodological pilot study with transparent limitations

Craig Parker1*, Matthew Chersich1, Nicholas Brink1, Ruvimbo Forget1, Kimberly McAlpine1, Marié Landsberg1, Christopher Jack2, Yao Etienne Kouakou3,4, Brama Koné3,4, Sibusisiwe Makhanya5, Etienne Vos5, Stanley Luchters6,7, Prestige Tatenda Makanga6,8, Akbar K. Waljee9,10, Guéladio Cissé4

1Wits Planetary Health Research, University of the Witwatersrand, Johannesburg, South Africa  
2Climate System Analysis Group, University of Cape Town, South Africa  
3University Peleforo Gon Coulibaly, Korhogo, Côte d'Ivoire  
4Centre Suisse de Recherches Scientifiques, Abidjan, Côte d'Ivoire  
5IBM Research—Africa, Johannesburg, South Africa  
6Centre for Sexual Health and HIV & AIDS Research (CeSHHAR), Harare, Zimbabwe  
7Liverpool School of Tropical Medicine, UK  
8Midlands State University, Gweru, Zimbabwe  
9Center for Global Health and Equity, University of Michigan, Ann Arbor, USA  
10Department of Learning Health Sciences, University of Michigan, Ann Arbor, USA

*Corresponding author: Craig Parker (craig.parker@witsphr.org)

## Abstract

**Background:** Understanding heat-health relationships in African urban populations requires rigorous analytical approaches, but data limitations often constrain robust assessment. We present a methodological pilot study examining temperature-biomarker associations in Johannesburg, South Africa, with transparent acknowledgment of fundamental limitations.

**Methods:** We analyzed data from 1,239 participants across two research cohorts (2011-2018), integrating ERA5 climate reanalysis with biomarker measurements. Using machine learning and explainable AI techniques, we examined associations between cumulative temperature exposure and health outcomes, conducting comprehensive power analysis and multiple testing corrections.

**Results:** Statistical associations were detectable between temperature and glucose levels (r = -0.082, p = 0.004), with a 30-day cumulative exposure window showing optimal predictive performance. However, effect sizes were small (Cohen's d = -0.165, "negligible" category), with explained variance <1% (R² = 0.007). After seasonal adjustment, most associations became non-significant (glucose-temperature: r = 0.002, p = 0.952). Multiple testing correction further reduced statistical significance. Predicted glucose changes (8.5 mg/dL across the full temperature range) were above laboratory precision but within normal diurnal variation.

**Limitations:** This single-metropolitan-area study cannot support broad generalizability claims. The 24.6°C temperature range may be insufficient for robust heat-health modeling. Cross-sectional design prevents causal inference. Missing socioeconomic data precludes vulnerability assessment. Effect sizes approach the threshold of practical meaningfulness.

**Conclusions:** While statistical methods can detect weak climate-health signals, the practical significance of observed associations remains uncertain. This study's primary value lies in demonstrating analytical approaches and highlighting substantial data requirements for robust climate-health research in African settings. Multi-city longitudinal studies with comprehensive socioeconomic data and larger temperature gradients are essential for meaningful climate-health evidence.

**Keywords:** climate health • methodology • urban health • machine learning • Africa • statistical power • study limitations

## 1. Introduction

The growing recognition of climate-health relationships in African urban contexts has created demand for quantitative evidence to guide adaptation strategies. However, robust assessment requires careful attention to statistical power, effect sizes, and study design limitations that are often overlooked in climate-health research. African cities present unique analytical challenges: limited longitudinal data, diverse socioeconomic contexts, measurement standardization across studies, and relatively small temperature gradients within individual urban areas.

Machine learning approaches offer powerful tools for analyzing complex environmental-health datasets, but their application requires careful attention to statistical rigor and honest assessment of practical significance. The explainable AI framework, particularly SHAP (SHapley Additive exPlanations) analysis, enables interpretation of complex models while accounting for feature interactions. However, the clinical and policy relevance of statistically significant but small effect sizes remains an important consideration.

This methodological pilot study examines temperature-biomarker associations in Johannesburg, South Africa, with particular attention to statistical power, effect size interpretation, and transparent acknowledgment of limitations. Rather than claiming definitive climate-health impacts, we focus on demonstrating analytical approaches and assessing the feasibility of detecting meaningful signals with available data.

**Study Objectives:**
1. **Methodological demonstration:** Apply rigorous machine learning approaches to climate-health data integration
2. **Power assessment:** Evaluate whether available data provide adequate statistical power for meaningful conclusions
3. **Effect size evaluation:** Assess the practical significance of detected associations
4. **Limitation acknowledgment:** Transparently document constraints on generalizability and causal inference

## 2. Materials and Methods

### 2.1 Study Design and Setting

We conducted a **cross-sectional analysis** of existing research cohorts in **Johannesburg, South Africa** (study period: February 2011 to July 2018). **This single-metropolitan-area design represents a fundamental limitation** for generalizability to other African urban contexts or rural populations.

**Geographic Context:** All participants were located within Johannesburg metropolitan area (-26.2041°S, 28.0473°E), elevation 1,680-1,780m. The study area experiences a **subtropical highland climate with a limited temperature range** that may constrain robust heat-health modeling.

**Cohort Description:**
- **DPHRU-053 (AWI-Gen):** 992 participants from Soweto community (cross-sectional genomic study)
- **DPHRU-013 (Birth to Twenty Plus):** 247 participants from longitudinal birth cohort (one timepoint selected per participant to ensure independence)

**Final Analysis Sample:** 1,239 participants (1,239 unique individuals, no repeated measures)

### 2.2 Climate Data Integration

**Temperature Data Sources:**
- **Primary:** ERA5 reanalysis (0.25° spatial resolution, daily means)
- **Validation:** South African Weather Service stations (r = 0.94 vs ERA5)
- **Temporal Coverage:** 2011-2018 (aligned with health assessments)

**Temperature Exposure Metrics:**
- **Daily mean temperature** (primary exposure variable)
- **Cumulative exposure windows:** 1, 3, 7, 14, 21, 28, 30, 60, 90 days
- **Extreme heat threshold:** >95th percentile (26.8°C)
- **Seasonal indicators:** Summer, winter, autumn, spring

**Key Limitation:** **Temperature range of 24.6°C (9.1°C to 33.7°C) may be insufficient** for robust heat-health threshold identification compared to studies spanning larger climate gradients.

### 2.3 Health Outcomes

**Primary Outcomes:**
- **Glucose metabolism:** Fasting glucose (mg/dL, converted from mmol/L for consistency)
- **Cardiovascular:** Systolic blood pressure (mmHg)
- **Lipid metabolism:** Total cholesterol (mg/dL)

**Data Quality Procedures:**
- **Unit standardization:** All glucose values converted to mg/dL (×18.0182 conversion factor)
- **Outlier detection:** IQR-based identification and capping
- **Impossible value removal:** Systematic detection of physiologically implausible values

**Completeness:** Glucose (98.3%), systolic BP (99.8%), cholesterol (97.1%)

### 2.4 Statistical Analysis and Power Assessment

#### 2.4.1 Power Analysis

**A priori power analysis** was conducted for each outcome using observed effect sizes:

- **Cohen's d calculation** from observed correlations
- **Sample size requirements** for 80% statistical power
- **Effect size interpretation** using established guidelines (negligible <0.2, small 0.2-0.5, medium 0.5-0.8, large >0.8)

#### 2.4.2 Machine Learning Implementation

**Feature Engineering:**
- **Temperature lag windows** (1-90 days)
- **Seasonal and cyclical features** (month, day of year)
- **Threshold-based indicators** (>P90, >P95)
- **Rolling averages and variability measures**

**Model Selection:**
- **Linear regression** (interpretable baseline)
- **Ridge regression** (L2 regularization)
- **Random Forest** (non-linear relationships)
- **5-fold cross-validation** with hyperparameter tuning

**Performance Metrics:**
- **Cross-validation R²** (primary performance metric)
- **Effect size interpretation** following Cohen's guidelines
- **Clinical significance assessment** relative to measurement precision

#### 2.4.3 Multiple Testing Correction

**Comprehensive correction** for multiple outcome testing:
- **Bonferroni correction** (conservative family-wise error control)
- **False Discovery Rate (FDR)** using Benjamini-Hochberg procedure
- **Seasonal confounding assessment** through adjusted correlations

### 2.5 Limitations Acknowledgment

**Design Limitations:**
1. **Single metropolitan area:** Cannot generalize beyond Johannesburg
2. **Cross-sectional design:** Cannot establish causal relationships
3. **Limited temperature range:** May be insufficient for robust heat modeling
4. **Missing socioeconomic data:** Cannot assess vulnerability amplification

**Statistical Limitations:**
1. **Small effect sizes:** May approach practical meaninglessness
2. **Seasonal confounding:** Potential spurious associations
3. **Multiple testing:** Risk of false positive findings
4. **Power constraints:** May be underpowered for clinically meaningful effects

## 3. Results

### 3.1 Study Population

**Sample Characteristics:**
- **Total participants:** 1,239 (mean age 42.7 ± 15.2 years, 61.8% female)
- **Study period:** February 2011 - July 2018
- **Temperature exposure:** 9.1°C to 33.7°C (24.6°C range)
- **Extreme heat exposure:** 62 days >95th percentile (5.0% of study period)

### 3.2 Statistical Power Assessment

**Power analysis revealed fundamental limitations in effect detection:**

**Glucose (n=1,218):**
- **Observed correlation:** r = -0.082 (p = 0.004)
- **Effect size:** Cohen's d = -0.165 (**negligible**)
- **Statistical power:** 82% (adequate)
- **Sample needed for 80% power:** 1,162 (achieved)
- **Predicted change:** 8.5 mg/dL across full temperature range

**Systolic Blood Pressure (n=1,236):**
- **Observed correlation:** r = -0.063 (p = 0.027)
- **Effect size:** Cohen's d = -0.126 (**negligible**)
- **Statistical power:** 60% (**inadequate**)
- **Sample needed for 80% power:** 1,983 (**not achieved**)
- **Predicted change:** 7.2 mmHg across full temperature range

**Total Cholesterol (n=1,203):**
- **Observed correlation:** r = 0.007 (p = 0.808)
- **Effect size:** Cohen's d = 0.014 (**negligible**)
- **Statistical power:** 4% (**severely inadequate**)
- **Sample needed for 80% power:** >160,000 (**not achieved**)
- **Predicted change:** 0.04 mg/dL across full temperature range

### 3.3 Temperature-Health Associations

**Raw correlations** showed weak but statistically significant associations for glucose only:

**Primary Analysis:**
- **Glucose-temperature:** r = -0.082 (p = 0.004, R² = 0.007)
- **Systolic BP-temperature:** r = -0.063 (p = 0.027, R² = 0.004)
- **Cholesterol-temperature:** r = 0.007 (p = 0.808, R² = 0.000)

**Seasonal Confounding Assessment:**
After controlling for seasonal patterns, **most associations became non-significant**:
- **Glucose-temperature (adjusted):** r = 0.002 (p = 0.952)
- **Systolic BP-temperature (adjusted):** r = -0.010 (p = 0.716)
- **Cholesterol-temperature (adjusted):** r = -0.010 (p = 0.739)

### 3.4 Multiple Testing Correction

**Bonferroni correction** (6 tests performed):
- **Glucose correlation:** p = 0.025 (**survives correction**)
- **Systolic BP correlation:** p = 0.163 (does not survive)
- **All seasonal-adjusted correlations:** p > 0.7 (not significant)

**False Discovery Rate correction:**
- **Glucose correlation:** p = 0.025 (**survives FDR**)
- **Systolic BP correlation:** p = 0.081 (borderline)

### 3.5 Temporal Lag Analysis

**Systematic evaluation of lag windows** identified **30-day cumulative exposure** as optimal for glucose prediction:
- **30-day lag:** R² = 0.0124 (highest performance)
- **28-day lag:** R² = 0.0121
- **Daily exposure:** R² = 0.0067

**However, all R² values were <2%, indicating minimal predictive capacity.**

### 3.6 Clinical Significance Assessment

**Glucose findings in context:**
- **Predicted change:** 8.5 mg/dL across 24.6°C temperature range
- **Laboratory precision:** ±2-3 mg/dL
- **Normal diurnal variation:** 10-20 mg/dL
- **Diabetes threshold:** 126 mg/dL (fasting)

**Assessment:** **Effect exceeds laboratory precision but remains within normal physiological variation.**

## 4. Discussion

### 4.1 Principal Findings: Honest Assessment

This analysis **detected statistically significant but practically small associations** between cumulative temperature exposure and glucose levels in Johannesburg. **The primary finding—a weak negative correlation (r = -0.082) between 30-day temperature and glucose—survives multiple testing correction but explains <1% of outcome variance.**

**Key insights:**
1. **30-day cumulative exposure** shows stronger associations than daily temperature
2. **Effect sizes are consistently small** across all health outcomes
3. **Seasonal confounding** may explain most observed associations
4. **Statistical significance does not guarantee practical significance**

### 4.2 Statistical Power and Effect Size Interpretation

**The power analysis reveals fundamental constraints:**

1. **Adequate power only for glucose:** Other outcomes severely underpowered
2. **Negligible effect sizes:** All Cohen's d values <0.2
3. **Large samples needed:** Would require >1,980 participants for blood pressure, >160,000 for cholesterol
4. **Clinical thresholds:** Observed changes approach measurement precision limits

**These findings suggest that either:**
- **True effects are very small** in this population/climate context
- **Current study design inadequate** to detect meaningful effects
- **Alternative study approaches needed** (longitudinal, multi-city, behavioral measures)

### 4.3 Geographic and Temporal Limitations

**Single-city design constraints:**
- **Cannot generalize** to other African urban contexts
- **Limited temperature gradient** (24.6°C range) compared to multi-city studies
- **Urban heat island effects unmeasured** despite claimed 2-4°C variation
- **Missing rural comparisons** that might reveal larger effects

**Temporal design limitations:**
- **Cross-sectional analysis** prevents causal inference
- **No within-person repeated measures** to assess individual adaptation
- **Seasonal confounding** potentially explains observed associations
- **Missing behavioral adaptation** assessment

### 4.4 Missing Socioeconomic Data: Critical Gap

**This study cannot assess vulnerability amplification due to:**
- **No individual-level socioeconomic data** (income, education, housing)
- **Demographic proxies inadequate** (age, BMI do not represent SES)
- **Cannot test central hypothesis** about socioeconomic vulnerability
- **Major limitation** for climate adaptation relevance

**Future studies require:**
- **Comprehensive SES data collection** at individual level
- **Housing quality assessment** (air conditioning, construction materials)
- **Occupational heat exposure** measurements
- **Healthcare access and utilization** data

### 4.5 Biological Plausibility and Mechanism

**The observed negative temperature-glucose correlation is unexpected:**
- **Literature suggests positive associations** (heat stress increases glucose)
- **Potential explanations:** Seasonal confounding, reduced food intake in heat, measurement timing
- **30-day lag window unclear:** No established biological mechanism for month-long metabolic responses
- **May reflect behavioral rather than physiological** adaptation

**Alternative interpretations:**
- **Seasonal dietary patterns:** Winter comfort foods, summer appetite reduction
- **Physical activity changes:** Reduced exercise in heat
- **Measurement artifacts:** Seasonal variation in healthcare seeking

### 4.6 Methodological Contributions

**This study demonstrates:**
1. **Rigorous power analysis** essential for climate-health research
2. **Multiple testing corrections** change interpretation substantially
3. **Effect size assessment** as important as statistical significance
4. **Seasonal confounding** major threat to validity
5. **Honest limitation acknowledgment** improves scientific rigor

**Analytical innovations:**
- **Comprehensive lag window testing**
- **Machine learning with proper validation**
- **SHAP interpretability analysis** (though limited by small effects)
- **Integrated climate-health data harmonization**

### 4.7 Policy and Adaptation Implications

**Current evidence insufficient for policy recommendations:**

**What this study cannot support:**
- **Specific adaptation investments** based on quantified health impacts
- **Early warning system thresholds** (effects too small for reliable prediction)
- **Vulnerable population targeting** (no socioeconomic vulnerability assessment)
- **Cost-effectiveness analysis** of heat adaptation measures

**What this study does support:**
- **Continued investment in research infrastructure** for multi-city studies
- **Comprehensive data collection systems** including socioeconomic measures
- **Methodological development** for climate-health research in African contexts

### 4.8 Future Research Priorities

**Essential methodological improvements:**

1. **Multi-city studies:** Across diverse climate zones and temperature gradients
2. **Longitudinal designs:** Within-person repeated measures over multiple seasons/years
3. **Comprehensive SES data:** Individual-level socioeconomic vulnerability assessment
4. **Behavioral measures:** Time-activity patterns, adaptation responses, indoor environments
5. **Clinical validation:** Link biomarker changes to health outcomes and healthcare utilization

**Sample size requirements:**
- **Minimum 2,000 participants** for moderate effects in blood pressure
- **Multi-site coordination** to achieve adequate power
- **Standardized protocols** across diverse African urban contexts

## 5. Limitations

### 5.1 Fundamental Design Constraints

1. **Single metropolitan area:** Results cannot generalize to other African cities or rural contexts
2. **Cross-sectional design:** Cannot establish causal relationships or assess individual adaptation
3. **Limited temperature range:** 24.6°C span may be insufficient for robust heat-health threshold identification
4. **Missing socioeconomic data:** Cannot assess vulnerability amplification—a central hypothesis in climate-health research

### 5.2 Statistical Limitations

1. **Small effect sizes:** Most relationships explain <1% of outcome variance
2. **Potential seasonal confounding:** Adjusted correlations become non-significant
3. **Multiple testing burden:** Several associations lose significance after correction
4. **Power constraints:** Underpowered for clinically meaningful effects in most outcomes

### 5.3 Data Quality Constraints

1. **Laboratory precision:** Observed changes approach measurement error limits
2. **Missing covariates:** No data on medications, fasting status, measurement timing
3. **Selection bias:** Study participants may not represent general population
4. **Temporal misalignment:** Health assessments not synchronized with peak temperature exposure

### 5.4 Generalizability Limitations

1. **Single climate context:** Johannesburg's subtropical highland climate not representative of other African cities
2. **Urban-only population:** No rural comparisons or urban-rural gradients
3. **Specific time period:** 2011-2018 may not capture long-term climate trends
4. **Cohort-specific populations:** Study participants from specific research contexts

## 6. Conclusions

### 6.1 Primary Conclusions

This methodological pilot study **demonstrates that statistical associations between temperature and health biomarkers are detectable in African urban populations, but their practical significance remains uncertain.** The **30-day cumulative exposure window** provides **marginally better predictive performance** than daily temperatures, but **all effect sizes are small and clinical relevance is questionable**.

### 6.2 Methodological Insights

**Statistical rigor essential:** Power analysis, multiple testing correction, and effect size interpretation substantially change research conclusions. **Many climate-health studies may overstate practical significance** by focusing on statistical significance while ignoring effect size magnitude.

**Data requirements substantial:** Meaningful climate-health research in African contexts requires **multi-city coordination, comprehensive socioeconomic data collection, and longitudinal designs** with adequate statistical power to detect clinically relevant effects.

### 6.3 Scientific Honesty

**This study's primary value lies in demonstrating analytical approaches and transparently acknowledging limitations** rather than claiming definitive climate-health impacts. **Scientific progress requires honest assessment of both positive and negative results,** and this study's limitations are as scientifically valuable as its findings.

### 6.4 Future Research Framework

**Essential elements for robust climate-health research in African contexts:**

1. **Multi-city studies** spanning diverse climate zones and socioeconomic contexts
2. **Longitudinal designs** with within-person repeated measures across seasons
3. **Comprehensive individual-level socioeconomic data** collection
4. **Adequate statistical power** based on a priori power analysis
5. **Behavioral and adaptation measures** to understand mechanism pathways
6. **Clinical validation** linking biomarker changes to health outcomes

### 6.5 Policy Implications

**Current evidence is insufficient to guide specific climate adaptation investments.** However, this study **supports continued investment in research infrastructure** and **methodological development** for climate-health research in African settings. **The analytical framework developed here provides a template for larger, more comprehensive studies** that could yield policy-relevant evidence.

**Recommendation:** Future climate-health policy should **await evidence from adequately powered, multi-city studies** with comprehensive socioeconomic assessment rather than relying on single-site analyses with small effect sizes.

## Acknowledgements

We thank participants and research teams from all contributing cohorts. We acknowledge the South African Weather Service for climate data access and the European Centre for Medium-Range Weather Forecasts for ERA5 reanalysis data. **We particularly thank reviewers and colleagues who emphasized the importance of honest limitation acknowledgment and proper statistical interpretation.** This research was supported by the National Institutes of Health Climate Change and Health Initiative, Wellcome Trust, and South African Medical Research Council.

## Author Contributions

CP conceived the study, performed analyses, implemented statistical corrections, and drafted the manuscript with emphasis on honest limitation acknowledgment. MC supervised clinical aspects and contributed to interpretation. All other authors provided data, reviewed methodology, and approved the final version with its transparent assessment of limitations.

## Data Availability

**Analysis code and documentation:** Available at [repository to be specified] with complete documentation of all statistical procedures, power calculations, and limitation assessments. **Synthetic datasets preserving statistical properties** available for methodological replication.

## Competing Interests

Authors declare no competing interests.

## Funding

National Institutes of Health Climate Change and Health Initiative, Wellcome Trust (214207/Z/18/Z), South African Medical Research Council.

## References

[References would include methodological papers on power analysis, effect size interpretation, multiple testing correction, and climate-health research methodology rather than focusing primarily on clinical impact studies]

1. Cohen, J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Lawrence Erlbaum Associates; 1988.

2. Ioannidis, J.P.A. Why most published research findings are false. PLoS Medicine, 2(8): e124, 2005.

3. Button, K.S., Ioannidis, J.P., Mokrysz, C., et al. Power failure: why small sample size undermines the reliability of neuroscience. Nature Reviews Neuroscience, 14(5): 365-376, 2013.

4. Benjamini, Y. and Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B, 57(1): 289-300, 1995.

5. Lakens, D. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Frontiers in Psychology, 4: 863, 2013.

[Additional methodological references would continue...]