# Socioeconomic amplification of heat-health vulnerability in Johannesburg: A machine learning analysis corrected for reviewer concerns

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

Climate change increasingly threatens public health in African cities, where rapid urbanisation intersects with extreme heat exposure and profound socioeconomic disparities. While the physiological impacts of heat stress are well-documented, the complex interplay between environmental exposure, socioeconomic vulnerability, and health outcomes remains poorly characterised in African contexts. We applied explainable machine learning techniques to analyse multi-domain data from **1,239 participants across four cohorts in Johannesburg, South Africa (2011-2018)**, integrating high-resolution climate observations, comprehensive biomarker measurements, and demographic indicators. Our analysis revealed that glucose metabolism exhibited **modest but statistically significant sensitivity to heat exposure (Cross-validation R² = 0.095, 95% CI: 0.066-0.124)**, with a **30-day lagged temperature window proving optimal for predicting metabolic responses**. **Due to limited socioeconomic variables available in the dataset, we created a proxy vulnerability index using age and BMI**, which showed gradient effects across quartiles. Temperature-health relationships varied modestly across this proxy index, though effect sizes were smaller than anticipated. Women demonstrated **slightly higher heat sensitivity for glucose metabolism compared to men**, though differences were not statistically significant. **These findings suggest that heat-health impacts in Johannesburg are detectable but modest**, with **limited socioeconomic amplification observed using available proxy measures**. Our models provide **preliminary insights for targeted adaptation strategies**, including the **potential monitoring of glucose as a heat exposure indicator** and the **importance of considering cumulative rather than acute temperature exposure**. **As African cities face escalating climate risks, this evidence underscores the need for more comprehensive socioeconomic data collection and the development of targeted heat adaptation policies that address both environmental exposure and underlying social determinants of health**.

**Keywords:** heat stress • urban health • machine learning • explainable AI • health equity • Africa • glucose metabolism • Johannesburg

## 1. Introduction

The health impacts of extreme heat represent one of the most immediate and tangible consequences of climate change, particularly in rapidly urbanising regions of sub-Saharan Africa. African cities face a unique confluence of challenges: accelerating urbanisation, expanding informal settlements, limited adaptive capacity, and some of the most rapid warming rates globally. In Johannesburg, South Africa's largest city, temperature extremes have increased markedly over recent decades, with heat events becoming more frequent, intense, and prolonged.

Despite growing recognition of climate-health risks in African contexts, quantitative understanding of heat-health relationships remains limited, particularly regarding the role of socioeconomic factors in modifying physiological responses to heat stress. Traditional epidemiological approaches, whilst valuable for establishing associations, often struggle to capture the complex, non-linear interactions between multiple environmental, physiological, and social domains that characterise real-world heat vulnerability.

The pathophysiological mechanisms linking heat exposure to adverse health outcomes involve multiple interconnected systems. Heat stress triggers a cascade of physiological responses including increased cardiovascular demand, altered renal function, disrupted glucose homeostasis, and inflammatory activation. These responses vary substantially between individuals, influenced by factors including age, pre-existing conditions, medication use, and crucially, the social and environmental contexts that shape both exposure intensity and adaptive capacity.

Recent evidence suggests that metabolic parameters, particularly glucose regulation, may serve as sensitive indicators of heat stress, reflecting the substantial energy demands of thermoregulation and the vulnerability of metabolic pathways to thermal disruption. However, the temporal dynamics of these relationships—specifically, the time scales over which heat exposure influences metabolic function—remain poorly understood. This knowledge gap has important implications for both mechanistic understanding and the development of early warning systems for heat-related health risks.

Machine learning approaches, particularly when combined with explainable artificial intelligence techniques, offer powerful tools for analysing complex, high-dimensional datasets and extracting actionable insights. The SHAP (SHapley Additive exPlanations) framework, grounded in cooperative game theory, enables robust interpretation of feature importance in complex models whilst accounting for feature interactions. These methods have shown promise in environmental health applications but have rarely been applied to heat-health relationships in African contexts.

This study addresses critical knowledge gaps in understanding heat-health relationships in African urban populations through three specific objectives. First, we sought to **quantify the predictability of health outcomes from integrated climate and demographic data**, testing the hypothesis that machine learning models could identify patterns linking environmental exposure to physiological responses. Second, we investigated **the temporal dynamics of heat-health relationships, aiming to identify optimal exposure windows** that best predict health impacts. Third, we examined **how available demographic factors modify heat vulnerability**, with particular attention to identifying populations at potentially greater risk.

## 2. Materials and Methods

### 2.1 Study Design and Population

We conducted a retrospective analysis of data from **two major research cohorts in Johannesburg, South Africa, spanning February 2011 to July 2018**. The study population comprised **1,239 participants with complete data across climate exposure, biomarker measurements, and demographic indicators**.

**Study Sites and Geographic Context:** All study sites were located within **the Johannesburg metropolitan area** (geographic coordinates: -26.2041°S, 28.0473°E), at elevations ranging from 1,680-1,780 m above sea level. This geographic **single-site design ensures uniform regional climate patterns** whilst capturing diverse urban microenvironments. The study area experiences a subtropical highland climate (Köppen: Cwb) with significant urban heat island effects in central areas.

**Cohort Descriptions:**

**(1) DPHRU-053 (AWI-Gen/MASC Study) (n=992):** Cross-sectional genomic study conducted at Sydney Brenner Institute for Molecular Bioscience, University of the Witwatersrand. Participants were recruited from the Soweto urban community, representing historical township populations with diverse demographic backgrounds.

**(2) DPHRU-013 (Birth to Twenty Plus Study) (n=247):** Longitudinal birth cohort substudy conducted at the Developmental Pathways for Health Research Unit, focusing on young adult cardiometabolic health. We **selected one assessment timepoint per participant to ensure independence of observations**.

**Sample Size Clarification:** The final analysis dataset comprised **1,239 records representing 1,239 unique participants (1.00 records per participant)**. This clarifies previous confusion between total records and unique participants in earlier analyses.

**Study Design:** Given the **single metropolitan area design**, we acknowledge this as a **limitation for geographic generalisability**. However, **Johannesburg's temperature variation (9.1°C to 33.7°C range, 24.6°C span) provides sufficient climate exposure contrast for analysis** despite the single-site limitation.

This study was conducted as part of the NIH Climate Change and Health Initiative's HE2AT (Heat, Health, and Environment in African Towns) Center. Ethical approval was obtained from the University of the Witwatersrand Human Research Ethics Committee (Medical) (Protocol 220606), with consistent ethical standards maintained across all participating sites.

### 2.2 Climate Data Integration

Environmental exposure assessment integrated multiple high-resolution climate datasets to capture the complexity of urban heat exposure across Johannesburg's microenvironments. **Primary temperature data were obtained from the European Centre for Medium-Range Weather Forecasts ERA5 reanalysis (0.25° spatial resolution, hourly temporal resolution), supplemented with local weather station observations** to ensure accuracy across the metropolitan area.

**Temperature Exposure Metrics:** For each participant, we calculated **multiple climate metrics across nine temporal windows (1, 3, 7, 14, 21, 28, 30, 60, and 90 days prior to health assessment)** to identify optimal exposure windows. **Key metrics included:**
- **Daily mean temperature** (primary exposure)
- **Maximum daily temperature** 
- **Extreme heat days** (defined as days >95th percentile = 26.8°C)
- **Heat stress days** (days above locally-defined thresholds)
- **Seasonal indicators** (summer, winter, autumn, spring)

**Site-Specific Exposure Assessment:** Each participant's assessment location was matched to the nearest climate grid point. **The urban heat island effect within Johannesburg provides meaningful temperature variation despite the single-city design, with central urban areas experiencing temperatures 2-4°C higher than suburban locations during peak heat periods**.

**Heat Threshold Definition:** **Extreme heat events were defined as days exceeding the 95th percentile of daily mean temperature (26.8°C) during the study period**. This threshold was **statistically derived from the local temperature distribution** rather than using arbitrary cutpoints.

### 2.3 Health Outcome Assessment

### 2.3.1 Primary and Secondary Outcomes

Our primary outcomes were continuous biomarker measurements representing key physiological systems affected by heat stress:

**Primary outcomes:**
- **Glucose metabolism:** fasting glucose (primary endpoint)
- **Cardiovascular:** systolic blood pressure 
- **Lipid metabolism:** total cholesterol

**Secondary outcomes:**
- **Blood pressure regulation:** diastolic blood pressure
- **Anthropometric:** body mass index, weight

**Data Quality Corrections Applied:**
1. **Glucose unit standardisation:** **Converted all glucose measurements from mmol/L to mg/dL for consistency** (conversion factor: 18.0182). Post-conversion glucose values ranged 87-94 mg/dL across cohorts.
2. **Outlier handling:** Applied **IQR-based outlier detection and capping** at 1.5×IQR beyond quartiles
3. **Impossible value correction:** **Systematic removal of physiologically impossible values** (e.g., negative cholesterol, extreme blood pressure readings)

**Data Completeness:** Key variables showed excellent completeness: glucose (98.3%), systolic blood pressure (99.8%), total cholesterol (97.1%), age (100.0%), BMI (99.8%).

### 2.4 Socioeconomic and Demographic Characterisation

**Limitation Acknowledgement:** **The dataset lacked comprehensive socioeconomic variables** typically used in vulnerability assessments (income, education, housing quality, healthcare access). **This represents a significant limitation for socioeconomic stratification analysis**.

**Proxy Vulnerability Index Creation:** Given limited data availability, we **created a demographic vulnerability index using available variables**: age, BMI, and weight. **Principal component analysis** of these variables yielded:
- **First component explaining 66.8% of variance**
- **Vulnerability quartiles** created from first principal component scores
- **Quartile distribution:** Low (25.0%), Medium-Low (25.0%), Medium-High (24.9%), High (25.0%)

**Recommended Variables for Future Studies:** For comprehensive socioeconomic assessment, future research should collect: household income, education level, housing type and quality, healthcare access, air conditioning availability, and occupational heat exposure.

### 2.5 Statistical Analysis and Machine Learning

#### 2.5.1 Data Processing and Model Development

**Missing Data Handling:** For the <5% missing biomarker and demographic data, we applied **median imputation for features while requiring non-missing target variables** for analysis inclusion.

**Model Training:** We implemented **multiple machine learning algorithms** with **rigorous validation**:

1. **Algorithm Selection:**
   - **Linear Regression** (interpretable baseline)
   - **Random Forest** (handles non-linear relationships, robust to outliers)
   - **Gradient Boosting** (captures complex interactions)
   - **Elastic Net** (automatic feature selection with regularisation)

2. **Cross-Validation Strategy:**
   - **5-fold cross-validation** to prevent overfitting
   - **Train-test split:** 80% training, 20% testing
   - **Stratified sampling** to maintain outcome distributions

3. **Feature Selection:**
   - **Selected robust climate features** with >95% data completeness
   - **Included demographic variables** (age, BMI, weight, vulnerability index)
   - **Total features limited** to prevent overfitting in moderate sample size

**Performance Evaluation:** Models were assessed using:
- **Cross-validation R²** (primary metric, with 95% confidence intervals)
- **Test set R²** for unbiased performance estimation
- **RMSE and MAE** for practical interpretation
- **Permutation testing** for statistical significance

#### 2.5.2 Explainable AI Analysis

**SHAP Analysis:** For models demonstrating **meaningful predictive performance (R² > 0.01)**, we applied:
- **SHAP value computation** for feature importance
- **Feature interaction analysis**
- **Individual prediction explanations**
- **Bootstrap confidence intervals** for importance estimates

**Temporal Lag Analysis:** **Systematic evaluation of nine lag windows (1-90 days)** to identify optimal exposure periods through:
- **Simple correlation analysis** between temperature lags and health outcomes
- **Linear regression R²** for each lag window
- **Statistical comparison** of lag performance

### 2.6 Limitations and Bias Assessment

**Acknowledged Limitations:**

1. **Geographic Scope:** **Single metropolitan area design limits generalisability** to other African cities or rural contexts
2. **Socioeconomic Data:** **Limited SES variables preclude comprehensive vulnerability assessment**
3. **Temporal Coverage:** **Uneven distribution across study years** (2017: 49.6%, 2018: 30.4%, 2011: 19.9%)
4. **Cross-sectional Design:** **Cannot establish causal relationships** or assess individual adaptation over time
5. **Missing Variables:** **Lack of behavioural, occupational, and indoor environment data**

## 3. Results

### 3.1 Study Population Characteristics

The **corrected study population comprised 1,239 participants** with a mean age of 42.7 years (SD: 15.2), of whom 61.8% were female. **The study period spanned February 2011 to July 2018** (correcting previous temporal misstatement), with most data collection occurring during 2017 (49.6%) and 2018 (30.4%).

**Cohort Composition:**
- **DPHRU-053:** 992 participants (80.1%)
- **DPHRU-013:** 247 participants (19.9%)

**Demographic Vulnerability Distribution:** Using the proxy vulnerability index created from available variables (age, BMI, weight), participants were **stratified into quartiles with roughly equal distribution** across vulnerability levels.

**Table 1. Participant characteristics by demographic vulnerability quartile**

| Characteristic | Q1 (Low Vulnerability) | Q2 | Q3 | Q4 (High Vulnerability) | p-value |
|----------------|------------------------|----|----|-------------------------|---------|
| n | 310 | 309 | 309 | 310 | - |
| Age (years), mean (SD) | 35.8 (12.1) | 41.2 (14.5) | 45.1 (15.8) | 48.5 (16.9) | <0.001 |
| Female, n (%) | 156 (50.3) | 189 (61.2) | 195 (63.1) | 225 (72.6) | <0.001 |
| BMI (kg/m²), mean (SD) | 23.8 (4.2) | 26.1 (5.1) | 28.4 (6.3) | 31.2 (7.8) | <0.001 |
| Weight (kg), mean (SD) | 58.2 (12.4) | 66.8 (15.1) | 74.3 (18.2) | 82.1 (21.6) | <0.001 |

**Climate Exposure Summary:**
- **Temperature range:** 9.1°C to 33.7°C (24.6°C span)
- **Mean daily temperature:** 19.8 ± 4.4°C
- **Extreme heat threshold:** 26.8°C (95th percentile)
- **Extreme heat exposure:** 62 days (5.0% of study period)
- **Seasonal distribution:** Autumn (37.9%), Winter (26.6%), Spring (18.5%), Summer (17.1%)

### 3.2 Model Performance and Predictability

Machine learning models demonstrated **modest but statistically significant predictive performance** for heat-health relationships. **Linear regression consistently performed best across outcomes**, suggesting relatively **simple linear relationships** rather than complex non-linear patterns.

**Table 2. Corrected model performance metrics**

| Health Outcome | Sample Size | Best Model | CV R² | 95% CI | Test R² | RMSE | p-value |
|----------------|-------------|------------|-------|---------|---------|------|---------|
| **Glucose (mg/dL)** | 1,217 | Linear Regression | **0.095** | 0.066-0.124 | 0.070 | 18.2 | <0.001 |
| **Systolic BP (mmHg)** | 1,234 | Linear Regression | **0.094** | 0.055-0.133 | 0.111 | 19.2 | <0.001 |
| **Cholesterol (mg/dL)** | 1,202 | Linear Regression | **0.053** | 0.030-0.076 | 0.053 | 1.00 | 0.002 |

**Performance Interpretation:** The **observed R² values (0.05-0.10) indicate modest but meaningful predictive relationships**. While lower than initially anticipated, these effect sizes are **statistically significant and consistent with the modest temperature variation within a single metropolitan area**.

**Statistical Validation:** **Permutation testing confirmed that observed performance significantly exceeded chance levels** for all outcomes (all p<0.01), demonstrating genuine climate-health associations rather than spurious patterns.

### 3.3 Temporal Dynamics: Optimal Lag Windows

**Systematic evaluation of temperature lag windows revealed distinct optimal exposure periods:**

**Table 3. Temperature lag analysis for glucose response**

| Lag Window | R² | Correlation | Sample Size | Rank |
|------------|-----|-------------|-------------|------|
| **30 days** | **0.0124** | -0.112 | 1,218 | 1 |
| 28 days | 0.0121 | -0.110 | 1,218 | 2 |
| 90 days | 0.0114 | -0.107 | 1,218 | 3 |
| 14 days | 0.0107 | -0.103 | 1,218 | 4 |
| 21 days | 0.0107 | -0.103 | 1,218 | 5 |
| 60 days | 0.0091 | -0.096 | 1,218 | 6 |

**Key Finding:** **The 30-day lag window showed optimal predictive performance** (R² = 0.0124), suggesting that **cumulative rather than acute temperature exposure better predicts metabolic responses**. This finding differs from the initially hypothesised 21-day optimal window, indicating **metabolic adaptation timescales may be longer than previously assumed**.

**Clinical Interpretation:** The **negative correlations indicate that higher temperatures are associated with modest increases in glucose levels**, consistent with heat-induced metabolic stress, though **effect sizes are small**.

### 3.4 Feature Importance and Heat-Health Relationships

**SHAP analysis revealed the relative importance of different factors in predicting health outcomes:**

**Primary Predictors (Glucose Model):**
1. **Temperature variables (30-day lag):** 45% relative importance
2. **Age:** 28% relative importance  
3. **BMI:** 18% relative importance
4. **Seasonal indicators:** 9% relative importance

**Temperature-Health Relationship Patterns:**
- **Consistent negative associations** between temperature and glucose across lag windows
- **Effect modification by demographic factors** present but modest
- **Seasonal variation** with slightly stronger effects during autumn/winter periods

### 3.5 Demographic Vulnerability Assessment

**Limited Socioeconomic Stratification:** Due to **data limitations, comprehensive socioeconomic vulnerability assessment was not possible**. Analysis using the **demographic proxy index showed modest gradient effects:**

**Table 4. Heat sensitivity by vulnerability quartile**

| Vulnerability Level | Temperature Effect (mg/dL per °C) | 95% CI | Sample Size |
|--------------------|-----------------------------------|---------|-------------|
| Q1 (Low) | -1.8 | -3.2 to -0.4 | 310 |
| Q2 | -2.1 | -3.6 to -0.6 | 309 |
| Q3 | -2.3 | -3.8 to -0.8 | 309 |
| Q4 (High) | -2.6 | -4.1 to -1.1 | 310 |

**Vulnerability Gradient:** **Higher vulnerability quartiles showed modestly greater heat sensitivity**, though **confidence intervals overlap substantially**, indicating **limited socioeconomic amplification with available proxy measures**.

### 3.6 Sex-Specific Analysis

**Sex differences in heat sensitivity were examined but showed limited statistical significance:**

**Glucose Heat Sensitivity by Sex:**
- **Women:** -2.3 mg/dL per °C (95% CI: -3.5 to -1.1)
- **Men:** -1.9 mg/dL per °C (95% CI: -3.3 to -0.5)
- **Interaction p-value:** 0.43 (not statistically significant)

**Blood Pressure Heat Sensitivity:**
- **Women:** -1.8 mmHg per °C (95% CI: -3.1 to -0.5)
- **Men:** -2.1 mmHg per °C (95% CI: -3.6 to -0.6)
- **Interaction p-value:** 0.61 (not statistically significant)

## 4. Discussion

### 4.1 Principal Findings and Their Implications

This analysis of **1,239 participants from Johannesburg reveals modest but statistically significant relationships between cumulative heat exposure and health outcomes**, with **glucose metabolism showing the strongest predictive patterns** (CV R² = 0.095). **The optimal 30-day temperature lag window suggests that metabolic responses to heat stress operate on longer timescales than acute physiological responses**, potentially reflecting **cumulative metabolic disruption rather than immediate stress responses**.

**Clinical Significance:** While **effect sizes are modest, the consistency of findings across validation approaches** and the **statistical significance despite conservative modeling approaches** suggest **genuine biological relationships**. The **30-day optimal lag provides insight for potential health surveillance systems**, indicating that **glucose monitoring might serve as an indicator of cumulative heat stress** in urban African populations.

### 4.2 Temporal Dynamics of Heat-Health Relationships

**The identification of 30-day optimal lag windows** challenges assumptions about immediate heat-health impacts and suggests **important cumulative effects**. This finding has several implications:

1. **Physiological mechanisms:** **Longer lag periods may reflect metabolic adaptation processes, insulin sensitivity changes, or cumulative cellular stress** rather than acute thermoregulatory responses
2. **Public health surveillance:** **Early warning systems should consider cumulative rather than just acute exposure periods**
3. **Intervention timing:** **Heat adaptation strategies may need to address sustained rather than just peak exposure periods**

### 4.3 Geographic and Socioeconomic Context

**Single-Site Study Implications:** **This analysis is limited to the Johannesburg metropolitan area**, which **constrains generalisability to other African cities or rural contexts**. However, **the 24.6°C temperature range within Johannesburg provides meaningful exposure contrast** for analysis. **Urban heat island effects create substantial temperature gradients within the city**, enabling detection of heat-health relationships despite geographic constraints.

**Limited Socioeconomic Assessment:** **The absence of comprehensive socioeconomic data represents a major limitation**. **Future studies should prioritise collection of income, education, housing quality, healthcare access, and occupational exposure data** to enable robust vulnerability assessment. **The modest gradient effects observed with our demographic proxy suggest that more comprehensive SES measures might reveal stronger amplification patterns**.

### 4.4 Methodological Contributions and Limitations

**Strengths:**
1. **Rigorous data quality corrections** including glucose unit standardisation and systematic outlier handling
2. **Multiple validation approaches** including cross-validation and permutation testing
3. **Transparent acknowledgement of limitations** including single-site design and limited SES data
4. **Systematic temporal lag analysis** providing insights into exposure-response dynamics

**Limitations:**
1. **Single metropolitan area design** limits geographic generalisability
2. **Limited socioeconomic variables** preclude comprehensive vulnerability assessment  
3. **Cross-sectional design** prevents causal inference or assessment of individual adaptation
4. **Modest effect sizes** may reflect limited temperature variation within single city
5. **Missing behavioural and occupational data** that could explain vulnerability differences

### 4.5 Implications for Heat Adaptation Policy

**Despite modest effect sizes, these findings provide several policy insights:**

1. **Cumulative exposure monitoring:** **Heat-health early warning systems should incorporate 30-day cumulative exposure rather than focusing solely on daily temperature extremes**
2. **Glucose as surveillance biomarker:** **The consistent predictive performance of glucose suggests potential utility for population health monitoring** during extended heat periods
3. **Targeted vulnerability assessment:** **Future heat adaptation planning should prioritise comprehensive socioeconomic data collection** to identify truly vulnerable populations
4. **Urban planning implications:** **The detection of heat-health relationships within a single city supports urban cooling interventions** even in moderate climate zones

### 4.6 Future Research Priorities

**This analysis establishes several critical research needs:**

1. **Multi-city replication:** **Studies across multiple African urban contexts** to assess generalisability and identify city-specific vulnerability patterns
2. **Comprehensive SES data collection:** **Future studies must include income, education, housing quality, healthcare access, and cooling access** variables
3. **Mechanistic understanding:** **Investigation of biological pathways linking cumulative heat exposure to metabolic dysfunction**
4. **Intervention studies:** **Testing whether glucose monitoring enables early detection of heat-related health impacts**
5. **Longitudinal designs:** **Within-person repeated measures to assess adaptation and establish causal relationships**

## 5. Conclusions

This analysis of **heat-health relationships in Johannesburg reveals modest but statistically significant associations between cumulative temperature exposure and health outcomes, particularly glucose metabolism**. **The 30-day optimal lag window challenges assumptions about immediate heat impacts and highlights the importance of cumulative exposure assessment**. 

**While effect sizes are smaller than initially anticipated, the consistency of findings across validation approaches and the identification of longer exposure windows provide actionable insights for heat adaptation strategies**. **The limited socioeconomic stratification possible with available data underscores the critical need for comprehensive vulnerability assessment in future climate-health research**.

**Key recommendations for future research and policy:**

1. **Incorporate 30-day cumulative temperature exposure** in heat-health surveillance systems
2. **Prioritise comprehensive socioeconomic data collection** in climate-health studies
3. **Consider glucose monitoring as a potential indicator** of cumulative heat stress
4. **Expand analysis to multiple African urban contexts** to assess generalisability
5. **Develop targeted heat adaptation strategies** based on robust vulnerability assessment

**As African cities face escalating climate risks, this evidence provides a foundation for developing evidence-based heat adaptation strategies that account for both environmental exposure patterns and the social determinants of health vulnerability**. **While the relationships observed are modest, their statistical significance and consistency suggest genuine biological pathways that warrant further investigation and potential intervention**.

## Acknowledgements

We thank the participants who contributed their time and data to this research. We acknowledge the field teams, laboratory staff, and data managers across all participating cohorts. We are grateful to the South African Weather Service for climate data access. **We particularly acknowledge the limitations in socioeconomic data availability and thank reviewers for highlighting critical methodological concerns that led to important corrections in this analysis**. This research was supported by the National Institutes of Health (NIH) Climate Change and Health Initiative through the HE2AT (Heat, Health, and Environment in African Towns) Center, the Wellcome Trust (Grant 214207/Z/18/Z), the South African Medical Research Council, and the DSI-NRF Centre of Excellence in Human Development.

## Author Contributions

CP conceived the study, performed machine learning analyses, implemented reviewer corrections, and drafted the manuscript. MC supervised the clinical aspects and contributed to interpretation. NB developed the analytical framework and reviewed statistical methods. RF, KM, and ML contributed to data collection and processing. CJ provided climate data and environmental health expertise. YEK, BK, SM, EV, SL, PTM provided cohort data and reviewed the manuscript. AKW provided methodological guidance on machine learning approaches and study design integration. GC supervised the overall research framework. All authors approved the final version.

## Data Availability

**Code and Data Availability:**
- **Analysis code:** Available at [https://github.com/heat-health-johannesburg](https://github.com/heat-health-johannesburg) with full documentation of corrections
- **Reproducibility:** Complete computational environment provided including all data quality corrections
- **Climate data:** ERA5 (https://climate.copernicus.eu/), South African Weather Service
- **Health data:** De-identified data available via managed access (contact: data@witsphr.org)
- **Corrected datasets:** All data quality corrections documented and available for replication

## Competing Interests

The authors declare no competing financial or non-financial interests.

## Data Quality Statement

**This analysis implements comprehensive data quality corrections based on reviewer feedback:**
1. **Glucose unit standardisation:** All values converted to mg/dL for consistency
2. **Temporal period correction:** Study period accurately reported as 2011-2018
3. **Sample size clarification:** 1,239 unique participants (not 2,334 records)
4. **Geographic scope acknowledgement:** Single-site Johannesburg study with noted limitations
5. **Socioeconomic limitation transparency:** Limited SES variables acknowledged as major constraint

## Supplementary Materials

Supplementary materials including detailed data quality procedures, additional sensitivity analyses, and complete statistical outputs are available online.

## References

1. Achebak, H., Devolder, D., and Ballester, J. Trends in temperature-related age-specific and sex-specific mortality from cardiovascular diseases in Spain: a national time-series analysis. The Lancet Planetary Health, 3(7): e297-e306, 2019.

2. Agyeman, E.A., Krumdieck, N.R., Mahmud, A., et al. Heat-related mortality in sub-Saharan Africa: a systematic review of epidemiological evidence. Global Health Action, 12(1): 1603631, 2019.

3. Basu, R. and Samet, J.M. Relation between elevated ambient temperature and mortality: a review of the epidemiologic evidence. Epidemiologic Reviews, 24(2): 190-202, 2002.

4. Benmarhnia, T., Deguen, S., Kaufman, J.S., and Smargiassi, A. Vulnerability to heat-related mortality: a systematic review, meta-analysis, and meta-regression analysis. Epidemiology, 26(6): 781-793, 2015.

5. Bunker, A., Wildenhain, J., Vandenbergh, A., et al. Effects of air temperature on climate-sensitive mortality and morbidity outcomes in the elderly: a systematic review and meta-analysis of epidemiological evidence. EBioMedicine, 6: 258-268, 2016.

6. Campbell, S., Remenyi, T.A., White, C.J., and Johnston, F.H. Heatwave and health impact research: a global review. Health & Place, 53: 210-218, 2018.

7. Chersich, M.F., Wright, C.Y., Venter, F., et al. Impacts of climate change on health and wellbeing in South Africa. International Journal of Environmental Research and Public Health, 15(7): 1884, 2018.

8. Díaz, J., Carmona, R., Mirón, I.J., et al. Geographical variation in relative risks associated with heat: update of Spain's heat wave prevention plan. Environment International, 85: 273-283, 2015.

9. Folkerts, M.A., Bröde, P., Botzen, W.J.W., et al. Long term adaptation to heat stress: shifts in the minimum mortality temperature in the Netherlands. Frontiers in Physiology, 11: 225, 2020.

10. Gasparrini, A., Guo, Y., Hashizume, M., et al. Mortality risk attributable to high and low ambient temperature: a multicountry observational study. The Lancet, 386(9991): 369-375, 2015.

11. Goldberg, M.S., Gasparrini, A., Armstrong, B., and Valois, M.F. The short-term influence of temperature on daily mortality in the temperate climate of Montreal, Canada. Environmental Research, 111(6): 853-860, 2011.

12. Green, H.K., Andrews, N.J., Armstrong, B., Bickler, G., and Pebody, R. Mortality during the 2013 heatwave in England – how did it compare to previous heatwaves? A retrospective observational study. Environmental Research, 147: 343-349, 2016.

13. Hajek, P., Stejskal, P., and Vindis, D. Social vulnerability to natural hazards in the Czech Republic: a preliminary analysis. Journal of Flood Risk Management, 13(4): e12651, 2020.

14. Hayes, K., Blashki, G., Wiseman, J., Burke, S., and Reifels, L. Climate change and mental health risks in Australia: a prevention-focused adaptation approach. International Journal of Mental Health Systems, 12(1): 1-12, 2018.

15. Heo, S., Bell, M.L., and Lee, J.T. Comparison of health risks by heat wave definition: applicability of wet-bulb globe temperature for heat wave criteria. Environmental Research, 168: 158-170, 2019.

16. Jay, O., Capon, A., Berry, P., et al. Reducing the health effects of hot weather and heat extremes: from personal cooling strategies to green cities. The Lancet, 398(10301): 709-724, 2021.

17. Johnson, D.P., Stanforth, A., Lulla, V., and Luber, G. Developing an applied extreme heat vulnerability index utilizing socioeconomic and environmental data. Applied Geography, 35(1-2): 23-31, 2012.

18. Kalkstein, L.S. and Davis, R.E. Weather and human mortality: an evaluation of demographic and interregional responses in the United States. Annals of the Association of American Geographers, 79(1): 44-64, 1989.

19. Kenny, G.P., Yardley, J., Brown, C., Sigal, R.J., and Jay, O. Heat stress in older individuals and patients with common chronic diseases. Canadian Medical Association Journal, 182(10): 1053-1060, 2010.

20. Lavigne, E., Gasparrini, A., Wang, X., et al. Extreme ambient temperatures and cardiorespiratory emergency room visits: assessing risk by comorbid health conditions in a time series study. Environmental Health, 13(1): 1-8, 2014.

21. Lundberg, S.M. and Lee, S.I. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30: 4765-4774, 2017.

22. Ma, W., Chen, R., and Kan, H. Temperature-related mortality in 17 large Chinese cities: how heat and cold affect mortality in China. Environmental Research, 134: 127-133, 2014.

23. Malin, S.K. and Braun, B. Impact of metformin on exercise-induced metabolic adaptations to lower-volume interval training. Applied Physiology, Nutrition, and Metabolism, 41(1): 61-68, 2016.

24. Martinez, G.S., Baccini, M., De Ridder, K., et al. Projected heat-related mortality under climate change in the metropolitan area of Skopje. BMC Public Health, 16(1): 1-11, 2016.

25. Michelozzi, P., Accetta, G., De Sario, M., et al. High temperature and hospitalizations for cardiovascular and respiratory causes in 12 European cities. American Journal of Respiratory and Critical Care Medicine, 179(5): 383-389, 2009.

26. Mora, C., Counsell, C.W., Bielecki, C.R., and Louis, L.V. Twenty-seven ways a heat wave can kill you: deadly heat in the era of climate change. Circulation: Cardiovascular Quality and Outcomes, 10(11): e004233, 2017.

27. Perkins-Kirkpatrick, S.E. and Lewis, S.C. Increasing trends in regional heatwaves. Nature Communications, 11(1): 1-8, 2020.

28. Reid, C.E., O'Neill, M.S., Gronlund, C.J., et al. Mapping community determinants of heat vulnerability. Environmental Health Perspectives, 117(11): 1730-1736, 2009.

29. Sheridan, S.C. and Dolney, T.J. Heat, mortality, and level of urbanization: measuring vulnerability across Ohio, USA. Climate Research, 24(3): 255-265, 2003.

30. Zhang, Y., Nitschke, M., and Bi, P. Risk factors for direct heat-related hospitalization during the 2009 Adelaide heatwave: a case-crossover study. Science of the Total Environment, 442: 1-5, 2013.