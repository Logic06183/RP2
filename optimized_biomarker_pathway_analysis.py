#!/usr/bin/env python3
"""
Optimized Biomarker Pathway Analysis Pipeline
==============================================
Focuses on geographic and temporal alignment between clinical trial data
and socioeconomic/climate data to understand physiological pathways.
"""

import pandas as pd
import numpy as np
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import shap
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

class OptimizedBiomarkerPathwayAnalysis:
    """Optimized analysis focusing on biomarker pathways in environmental context."""
    
    def __init__(self):
        self.base_path = Path("/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized")
        self.data_path = self.base_path / "data"
        self.results_path = self.base_path / "results"
        self.figures_path = self.base_path / "figures"
        
        # Create directories
        self.results_path.mkdir(exist_ok=True)
        self.figures_path.mkdir(exist_ok=True)
        
        # Initialize storage
        self.clinical_data = None
        self.socio_data = None
        self.climate_data = None
        self.integrated_data = None
        self.pathway_models = {}
        self.shap_explainers = {}
        
    def load_and_analyze_clinical_geography(self):
        """Load clinical data and analyze geographic distribution."""
        print("\n" + "="*80)
        print("STEP 1: Loading Master Integrated Dataset")
        print("="*80)
        
        # Load the master integrated dataset instead of just clinical data
        master_path = self.data_path / "MASTER_INTEGRATED_DATASET.csv"
        if master_path.exists():
            print(f"Loading master integrated dataset: {master_path}")
            self.clinical_data = pd.read_csv(master_path)
            print(f"Loaded {len(self.clinical_data)} total records from master dataset")
        else:
            # Fallback to clinical data only
            clinical_path = self.data_path / "clinical/RP2/HEAT_Johannesburg_FINAL_20250811_163049.csv"
            self.clinical_data = pd.read_csv(clinical_path)
            print(f"Loaded {len(self.clinical_data)} clinical records (fallback)")
        
        print(f"Loaded {len(self.clinical_data)} clinical trial records")
        
        # Identify geographic hotspots
        geo_cols = ['latitude', 'longitude', 'ward', 'municipality', 'facility_name']
        available_geo = [col for col in geo_cols if col in self.clinical_data.columns]
        
        geographic_summary = {}
        for col in available_geo:
            if col in self.clinical_data.columns:
                non_null = self.clinical_data[col].notna().sum()
                unique = self.clinical_data[col].nunique()
                geographic_summary[col] = {
                    'available': non_null,
                    'unique_locations': unique,
                    'coverage': non_null / len(self.clinical_data)
                }
        
        # Identify temporal coverage
        date_cols = [col for col in self.clinical_data.columns if 'date' in col.lower()]
        if 'year' in self.clinical_data.columns:
            year_range = self.clinical_data['year'].dropna()
            temporal_range = {
                'min_year': int(year_range.min()),
                'max_year': int(year_range.max()),
                'unique_years': sorted(year_range.unique().astype(int).tolist())
            }
        else:
            temporal_range = {'status': 'No year column found'}
        
        # Analyze biomarker availability
        biomarkers = {
            'CD4 cell count (cells/µL)': 'cd4_count',
            'Hemoglobin (g/dL)': 'hemoglobin',
            'Creatinine (mg/dL)': 'creatinine',
            'HIV viral load (copies/mL)': 'viral_load',
            'systolic blood pressure': 'sys_bp',
            'diastolic blood pressure': 'dia_bp',
            'Body mass index': 'bmi',
            'Temperature': 'body_temp'
        }
        
        biomarker_coverage = {}
        for bio_name, bio_col in biomarkers.items():
            if bio_name in self.clinical_data.columns:
                non_null = self.clinical_data[bio_name].notna().sum()
                biomarker_coverage[bio_name] = {
                    'available': non_null,
                    'coverage': non_null / len(self.clinical_data),
                    'mean': self.clinical_data[bio_name].mean() if non_null > 0 else None,
                    'std': self.clinical_data[bio_name].std() if non_null > 0 else None
                }
        
        self.clinical_geography = {
            'total_records': len(self.clinical_data),
            'geographic_coverage': geographic_summary,
            'temporal_range': temporal_range,
            'biomarker_coverage': biomarker_coverage
        }
        
        print(f"\nGeographic Coverage:")
        for col, info in geographic_summary.items():
            print(f"  {col}: {info['unique_locations']} unique locations ({info['coverage']*100:.1f}% coverage)")
        
        print(f"\nTemporal Coverage: {temporal_range}")
        
        print(f"\nBiomarker Availability:")
        for bio, info in biomarker_coverage.items():
            if info['available'] > 0:
                print(f"  {bio}: {info['available']} records ({info['coverage']*100:.1f}%)")
        
        return self.clinical_geography
    
    def load_and_filter_socioeconomic_data(self):
        """Extract socioeconomic data from master dataset."""
        print("\n" + "="*80)
        print("STEP 2: Extracting Socioeconomic Data from Master Dataset")
        print("="*80)
        
        # Extract GCRO data from master dataset (data_source == 'GCRO')
        if 'data_source' in self.clinical_data.columns:
            self.socio_data = self.clinical_data[self.clinical_data['data_source'] == 'GCRO'].copy()
            print(f"Extracted {len(self.socio_data)} GCRO socioeconomic records from master dataset")
        else:
            # If no data_source column, check for GCRO-specific columns
            gcro_indicators = ['gcro_id', 'survey_wave', 'q1_19_4_heat']
            has_gcro_cols = any(col in self.clinical_data.columns for col in gcro_indicators)
            
            if has_gcro_cols:
                # Filter to rows that have GCRO data
                gcro_mask = self.clinical_data['gcro_id'].notna() if 'gcro_id' in self.clinical_data.columns else pd.Series([True] * len(self.clinical_data))
                self.socio_data = self.clinical_data[gcro_mask].copy()
                print(f"Identified {len(self.socio_data)} records with GCRO socioeconomic data")
            else:
                print("No GCRO data found in master dataset, using empty socioeconomic dataset")
                self.socio_data = pd.DataFrame()
        
        
        if len(self.socio_data) > 0:
            # Analyze temporal and geographic coverage
            if 'survey_year' in self.socio_data.columns:
                years_available = sorted(self.socio_data['survey_year'].unique())
                print(f"Survey years available: {years_available}")
            
            if 'ward' in self.socio_data.columns:
                wards_available = self.socio_data['ward'].nunique()
                print(f"Wards represented: {wards_available}")
            
            # Summarize key socioeconomic variables
            socio_vars = ['age', 'sex', 'race', 'education', 'employment', 'income']
            available_socio_vars = [var for var in socio_vars if var in self.socio_data.columns]
            print(f"Socioeconomic variables available: {available_socio_vars}")
            
            for var in available_socio_vars:
                coverage = self.socio_data[var].notna().sum()
                print(f"  {var}: {coverage} records ({coverage/len(self.socio_data)*100:.1f}% coverage)")
        
        return self.socio_data
    
    def harmonize_socioeconomic_variables(self):
        """Harmonize GCRO variable names across surveys."""
        # Common variable mappings
        variable_mappings = {
            'age': ['age', 'q2', 'Q2', 'AGE'],
            'sex': ['sex', 'gender', 'q1', 'Q1', 'SEX', 'GENDER'],
            'race': ['race', 'q3', 'Q3', 'RACE', 'population_group'],
            'education': ['education', 'educ', 'q6', 'Q6', 'highest_education'],
            'employment': ['employment', 'employ', 'q10', 'Q10', 'employment_status'],
            'income': ['income', 'household_income', 'q14', 'Q14', 'monthly_income']
        }
        
        for target_var, possible_names in variable_mappings.items():
            for col in self.socio_data.columns:
                if col in possible_names:
                    self.socio_data[target_var] = self.socio_data[col]
                    break
        
        print(f"Harmonized variables: {list(variable_mappings.keys())}")
    
    def load_climate_data(self):
        """Extract climate data from master dataset."""
        print("\n" + "="*80)
        print("STEP 3: Extracting Climate Data from Master Dataset")
        print("="*80)
        
        # Extract climate variables from master dataset
        climate_vars = [
            'era5_temp_1d_mean', 'era5_temp_1d_max', 
            'era5_temp_7d_mean', 'era5_temp_30d_mean',
            'era5_temp_30d_extreme_days', 'q1_19_4_heat'
        ]
        
        available_climate_vars = [var for var in climate_vars if var in self.clinical_data.columns]
        
        if available_climate_vars:
            # Extract climate data where variables are not null
            climate_mask = self.clinical_data[available_climate_vars].notna().any(axis=1)
            self.climate_data = self.clinical_data[climate_mask][available_climate_vars].copy()
            
            print(f"Extracted {len(self.climate_data)} records with climate data")
            print(f"Available climate variables: {available_climate_vars}")
            
            # Show coverage for each variable
            for var in available_climate_vars:
                coverage = self.clinical_data[var].notna().sum()
                print(f"  {var}: {coverage} records ({coverage/len(self.clinical_data)*100:.1f}% coverage)")
        else:
            print("Warning: No climate variables found in master dataset")
            self.climate_data = pd.DataFrame()
        
        return self.climate_data
    
    def create_balanced_integrated_dataset(self):
        """Create integrated dataset with spatial-temporal matching for biomarker analysis."""
        print("\n" + "="*80)
        print("STEP 4: Creating Spatially-Temporally Matched Dataset for Biomarker Analysis")
        print("="*80)
        
        # Separate clinical and GCRO data
        clinical_records = self.clinical_data[self.clinical_data['anonymous_patient_id'].notna()].copy()
        gcro_records = self.clinical_data[self.clinical_data['gcro_id'].notna()].copy()
        
        print(f"Clinical records: {len(clinical_records)}")
        print(f"GCRO records: {len(gcro_records)}")
        
        # Match climate and socioeconomic data to clinical records
        enhanced_clinical = self.match_environmental_data_to_clinical(clinical_records, gcro_records)
        
        # Use enhanced clinical data as the integrated dataset
        self.integrated_data = enhanced_clinical
        
        # Identify records with biomarkers
        biomarker_cols = [
            'CD4 cell count (cells/µL)',
            'Hemoglobin (g/dL)', 
            'Creatinine (mg/dL)',
            'HIV viral load (copies/mL)',
            'systolic blood pressure',
            'diastolic blood pressure'
        ]
        
        # Count records with each biomarker
        biomarker_counts = {}
        for bio in biomarker_cols:
            if bio in self.integrated_data.columns:
                count = self.integrated_data[bio].notna().sum()
                biomarker_counts[bio] = count
                print(f"  {bio}: {count} records with data")
        
        # Identify records with predictors (including matched variables)
        demographic_predictors = [
            'Age (at enrolment)', 'age', 'Sex', 'sex', 'Race', 'race'
        ]
        socioeconomic_predictors = [
            'education', 'Employment status', 'employment',
            'income', 'Personal income', 'Household income',
            'ward', 'municipality',
            'matched_education', 'matched_employment', 'matched_income', 'matched_municipality'
        ]
        climate_predictors = [
            'era5_temp_1d_mean', 'era5_temp_1d_max', 'era5_temp_7d_mean', 'era5_temp_30d_mean',
            'era5_temp_30d_extreme_days', 'q1_19_4_heat',
            'matched_era5_temp_1d_mean', 'matched_era5_temp_1d_max',
            'matched_era5_temp_7d_mean', 'matched_era5_temp_30d_mean',
            'matched_era5_temp_30d_extreme_days', 'matched_q1_19_4_heat'
        ]
        
        all_predictors = demographic_predictors + socioeconomic_predictors + climate_predictors
        available_predictors = [col for col in all_predictors if col in self.integrated_data.columns]
        print(f"\nAvailable predictor columns: {len(available_predictors)}")
        
        # Show coverage of matched variables
        matched_vars = [col for col in available_predictors if col.startswith('matched_')]
        if matched_vars:
            print(f"Matched environmental variables: {len(matched_vars)}")
            for var in matched_vars:
                count = self.integrated_data[var].notna().sum()
                print(f"  {var}: {count} records ({count/len(self.integrated_data)*100:.1f}% coverage)")
        
        # Focus on records that have both biomarkers AND predictors
        records_with_both = 0
        for bio in biomarker_counts.keys():
            if biomarker_counts[bio] > 0:
                bio_mask = self.integrated_data[bio].notna()
                predictor_mask = self.integrated_data[available_predictors].notna().any(axis=1)
                both_mask = bio_mask & predictor_mask
                count_both = both_mask.sum()
                records_with_both = max(records_with_both, count_both)
                
                # Count how many have matched environmental data
                if matched_vars:
                    matched_mask = self.integrated_data[matched_vars].notna().any(axis=1)
                    biomarker_with_matched = (bio_mask & matched_mask).sum()
                    print(f"  {bio}: {count_both} total, {biomarker_with_matched} with matched environmental data")
                else:
                    print(f"  {bio}: {count_both} records with both biomarker and predictor data")
        
        print(f"\nIntegrated Dataset Summary:")
        print(f"  Total records: {len(self.integrated_data)}")
        
        # Analyze data sources if available
        if 'data_source' in self.integrated_data.columns:
            source_counts = self.integrated_data['data_source'].value_counts()
            for source, count in source_counts.items():
                print(f"  {source}: {count} records")
        
        # Analyze clinical vs GCRO records
        clinical_records = self.integrated_data['anonymous_patient_id'].notna().sum()
        gcro_records = self.integrated_data['gcro_id'].notna().sum() if 'gcro_id' in self.integrated_data.columns else 0
        print(f"  Clinical records (with patient ID): {clinical_records}")
        print(f"  GCRO records (with GCRO ID): {gcro_records}")
        print(f"  Records suitable for biomarker modeling: {records_with_both}")
        print(f"  Total columns: {len(self.integrated_data.columns)}")
        
        # Save integrated dataset
        output_path = self.results_path / "biomarker_integrated_dataset.csv"
        self.integrated_data.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        
        return self.integrated_data
    
    def match_environmental_data_to_clinical(self, clinical_records, gcro_records):
        """Match environmental and socioeconomic data to clinical records using regional temporal matching."""
        print("\nPerforming regional temporal matching...")
        
        enhanced_clinical = clinical_records.copy()
        
        # Get the years from clinical records
        clinical_years = clinical_records['year'].dropna().unique()
        print(f"Clinical data years: {sorted(clinical_years.astype(int)) if len(clinical_years) > 0 else 'None'}")
        
        # Check data availability
        print(f"Clinical records with coordinates: {(clinical_records['latitude'].notna() & clinical_records['longitude'].notna()).sum()}/{len(clinical_records)}")
        print(f"GCRO records with ward info: {gcro_records['ward'].notna().sum()}/{len(gcro_records)}")
        
        # Variables to match from GCRO data
        variables_to_match = {
            'climate': ['era5_temp_1d_mean', 'era5_temp_1d_max', 'era5_temp_7d_mean', 'era5_temp_30d_mean', 
                       'era5_temp_30d_extreme_days', 'q1_19_4_heat'],
            'socioeconomic': ['education', 'employment', 'income', 'municipality']
        }
        
        # Initialize columns in enhanced_clinical
        for var_type, var_list in variables_to_match.items():
            for var in var_list:
                if var in gcro_records.columns:
                    enhanced_clinical[f'matched_{var}'] = np.nan
        
        # Create regional profiles by survey year
        regional_profiles = {}
        
        for year in gcro_records['survey_year'].dropna().unique():
            year_data = gcro_records[gcro_records['survey_year'] == year]
            
            profile = {}
            for var_type, var_list in variables_to_match.items():
                for var in var_list:
                    if var in year_data.columns:
                        values = year_data[var].dropna()
                        if len(values) > 0:
                            if var in ['education', 'employment', 'q1_19_4_heat', 'municipality']:  # Categorical
                                profile[var] = values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0]
                            else:  # Numerical
                                profile[var] = values.median()
            
            regional_profiles[year] = profile
        
        print(f"Created regional profiles for {len(regional_profiles)} survey years")
        
        matched_count = 0
        total_clinical = len(clinical_records)
        
        # Match each clinical record with appropriate regional profile
        for idx, clinical_row in clinical_records.iterrows():
            clinical_year = clinical_row.get('year', None)
            
            if pd.isna(clinical_year):
                continue
            
            # Find the closest survey year
            best_match_year = None
            min_year_diff = float('inf')
            
            for survey_year in regional_profiles.keys():
                year_diff = abs(survey_year - clinical_year)
                if year_diff < min_year_diff and year_diff <= 3:  # Allow ±3 years
                    min_year_diff = year_diff
                    best_match_year = survey_year
            
            if best_match_year is not None:
                # Apply regional profile to clinical record
                profile = regional_profiles[best_match_year]
                for var, value in profile.items():
                    enhanced_clinical.loc[idx, f'matched_{var}'] = value
                matched_count += 1
        
        # For unmatched records, use overall regional averages
        if matched_count < total_clinical:
            overall_profile = {}
            for var_type, var_list in variables_to_match.items():
                for var in var_list:
                    if var in gcro_records.columns:
                        values = gcro_records[var].dropna()
                        if len(values) > 0:
                            if var in ['education', 'employment', 'q1_19_4_heat', 'municipality']:  # Categorical
                                overall_profile[var] = values.mode().iloc[0] if len(values.mode()) > 0 else values.iloc[0]
                            else:  # Numerical
                                overall_profile[var] = values.median()
            
            # Apply to unmatched records
            for idx, clinical_row in clinical_records.iterrows():
                # Check if already matched
                already_matched = any(not pd.isna(enhanced_clinical.loc[idx, f'matched_{var}']) 
                                    for var_list in variables_to_match.values() 
                                    for var in var_list if f'matched_{var}' in enhanced_clinical.columns)
                
                if not already_matched:
                    for var, value in overall_profile.items():
                        enhanced_clinical.loc[idx, f'matched_{var}'] = value
                    matched_count += 1
        
        print(f"Successfully matched {matched_count}/{total_clinical} clinical records ({matched_count/total_clinical*100:.1f}%)")
        
        # Verify matched variables
        for var_type, var_list in variables_to_match.items():
            for var in var_list:
                matched_col = f'matched_{var}'
                if matched_col in enhanced_clinical.columns:
                    count = enhanced_clinical[matched_col].notna().sum()
                    if count > 0:
                        print(f"  {matched_col}: {count} records")
        
        # Add GCRO context data
        if len(gcro_records) > len(clinical_records) * 2:
            sampled_gcro = gcro_records.sample(n=len(clinical_records) * 2, random_state=42)
        else:
            sampled_gcro = gcro_records.copy()
            
        # Rename columns for consistency
        for var_type, var_list in variables_to_match.items():
            for var in var_list:
                if var in sampled_gcro.columns:
                    sampled_gcro = sampled_gcro.rename(columns={var: f'matched_{var}'})
        
        # Combine datasets
        final_dataset = pd.concat([enhanced_clinical, sampled_gcro], ignore_index=True)
        
        print(f"Final integrated dataset: {len(final_dataset)} records")
        print(f"  Clinical (with environmental matches): {len(enhanced_clinical)}")
        print(f"  Additional GCRO context: {len(sampled_gcro)}")
        
        return final_dataset
    
    def build_biomarker_pathway_models(self):
        """Build XAI models focusing on biomarker pathways."""
        print("\n" + "="*80)
        print("STEP 5: Building Biomarker Pathway Models with XAI")
        print("="*80)
        
        # Define biomarker targets
        biomarker_targets = [
            'CD4 cell count (cells/µL)',
            'Hemoglobin (g/dL)', 
            'Creatinine (mg/dL)',
            'HIV viral load (copies/mL)',
            'systolic blood pressure',
            'diastolic blood pressure'
        ]
        
        # Define predictor categories with correct column names including matched variables
        demographic_predictors = [
            'Age (at enrolment)', 'age',  # Clinical and GCRO versions
            'Sex', 'sex',
            'Race', 'race'
        ]
        socioeconomic_predictors = [
            'education', 'Employment status', 'employment',
            'income', 'Personal income', 'Household income',
            'ward', 'municipality',
            # Matched variables from spatial-temporal matching
            'matched_education', 'matched_employment', 'matched_income', 'matched_municipality'
        ]
        climate_predictors = [
            'era5_temp_1d_mean', 'era5_temp_1d_max', 
            'era5_temp_7d_mean', 'era5_temp_30d_mean',
            'era5_temp_30d_extreme_days', 'q1_19_4_heat',
            # Matched variables from spatial-temporal matching
            'matched_era5_temp_1d_mean', 'matched_era5_temp_1d_max',
            'matched_era5_temp_7d_mean', 'matched_era5_temp_30d_mean',
            'matched_era5_temp_30d_extreme_days', 'matched_q1_19_4_heat'
        ]
        
        all_predictors = demographic_predictors + socioeconomic_predictors + climate_predictors
        
        # Build models for each biomarker
        self.pathway_results = {}
        
        for biomarker in biomarker_targets:
            if biomarker not in self.integrated_data.columns:
                continue
            
            print(f"\nModeling {biomarker}...")
            
            # Get data for this biomarker
            biomarker_data = self.integrated_data[self.integrated_data[biomarker].notna()].copy()
            
            if len(biomarker_data) < 50:
                print(f"  Insufficient data for {biomarker} (n={len(biomarker_data)})")
                continue
            
            # Prepare features - prioritize clinical data columns first
            available_features = []
            for predictor in all_predictors:
                if predictor in biomarker_data.columns:
                    non_null_count = biomarker_data[predictor].notna().sum()
                    if non_null_count >= 10:  # Require at least 10 non-null values
                        available_features.append(predictor)
            
            # Remove duplicates (clinical vs GCRO versions of same variable)
            # Keep clinical versions when both exist
            deduped_features = []
            for feat in available_features:
                if feat == 'Age (at enrolment)' and 'age' in available_features:
                    if feat not in deduped_features:  # Prefer clinical version
                        deduped_features.append(feat)
                elif feat == 'Sex' and 'sex' in available_features:
                    if feat not in deduped_features:  # Prefer clinical version
                        deduped_features.append(feat)
                elif feat == 'Race' and 'race' in available_features:
                    if feat not in deduped_features:  # Prefer clinical version
                        deduped_features.append(feat)
                elif feat not in ['age', 'sex', 'race']:  # Keep other features
                    deduped_features.append(feat)
            
            available_features = deduped_features
            
            if len(available_features) < 2:
                print(f"  Insufficient features for {biomarker} (only {len(available_features)} available)")
                continue
            
            print(f"  Using {len(available_features)} features: {available_features}")
            
            X = biomarker_data[available_features].copy()
            y = biomarker_data[biomarker].copy()
            
            # Handle categorical variables more carefully
            categorical_encoders = {}
            categorical_vars = ['Sex', 'Race', 'sex', 'race', 'matched_education', 'matched_employment', 
                              'matched_q1_19_4_heat', 'matched_municipality', 'education', 'employment']
            
            for col in X.columns:
                if X[col].dtype == 'object' or col in categorical_vars:
                    le = LabelEncoder()
                    # Fill missing values with a specific category
                    X[col] = X[col].fillna('missing')
                    # Convert to string to ensure consistent encoding
                    X[col] = X[col].astype(str)
                    X[col] = le.fit_transform(X[col])
                    categorical_encoders[col] = le
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # SHAP analysis
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            
            # Calculate pathway importance scores
            pathway_importance = {
                'demographic': 0,
                'socioeconomic': 0,
                'climate': 0
            }
            
            feature_importance = {}
            for i, feature in enumerate(X.columns):
                importance = np.abs(shap_values.values[:, i]).mean()
                feature_importance[feature] = importance
                
                # Categorize features into pathways (including matched variables)
                if feature in demographic_predictors:
                    pathway_importance['demographic'] += importance
                elif feature in socioeconomic_predictors or 'matched_' in feature and any(svar in feature for svar in ['education', 'employment', 'income', 'municipality']):
                    pathway_importance['socioeconomic'] += importance
                elif feature in climate_predictors or 'matched_' in feature and any(cvar in feature for cvar in ['temp', 'heat', 'era5']):
                    pathway_importance['climate'] += importance
                else:
                    # If feature doesn't fit clear category, assign to most appropriate
                    if 'temp' in feature.lower() or 'climate' in feature.lower() or 'heat' in feature.lower():
                        pathway_importance['climate'] += importance
                    elif 'age' in feature.lower() or 'sex' in feature.lower() or 'race' in feature.lower():
                        pathway_importance['demographic'] += importance
                    else:
                        pathway_importance['socioeconomic'] += importance
            
            # Normalize pathway importance
            total_importance = sum(pathway_importance.values())
            if total_importance > 0:
                pathway_importance = {k: v/total_importance*100 for k, v in pathway_importance.items()}
            
            # Store results
            self.pathway_results[biomarker] = {
                'n_samples': len(biomarker_data),
                'n_features': len(available_features),
                'features_used': available_features,
                'model_performance': {
                    'r2': r2,
                    'rmse': rmse
                },
                'feature_importance': feature_importance,
                'pathway_importance': pathway_importance,
                'shap_values': shap_values
            }
            
            print(f"  Sample size: {len(biomarker_data)}")
            print(f"  R² = {r2:.3f}, RMSE = {rmse:.2f}")
            print(f"  Pathway importance:")
            print(f"    Demographic: {pathway_importance['demographic']:.1f}%")
            print(f"    Socioeconomic: {pathway_importance['socioeconomic']:.1f}%")
            print(f"    Climate: {pathway_importance['climate']:.1f}%")
            
            # Create SHAP plot
            self.create_shap_pathway_plot(biomarker, shap_values, X_test)
        
        return self.pathway_results
    
    def create_shap_pathway_plot(self, biomarker, shap_values, X_test):
        """Create SHAP plot highlighting pathway contributions."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Summary plot
        ax = axes[0]
        shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
        plt.sca(ax)
        ax.set_title(f'SHAP Feature Importance\n{biomarker}')
        
        # Pathway importance bar plot
        ax = axes[1]
        pathway_data = self.pathway_results[biomarker]['pathway_importance']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(pathway_data.keys(), pathway_data.values(), color=colors)
        ax.set_ylabel('Relative Importance (%)')
        ax.set_title(f'Pathway Contributions\n{biomarker}')
        ax.set_ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        safe_name = biomarker.replace('/', '_').replace(' ', '_')
        fig_path = self.figures_path / f"pathway_analysis_{safe_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved visualization: {fig_path.name}")
    
    def generate_physiological_insights(self):
        """Generate insights about physiological pathways."""
        print("\n" + "="*80)
        print("STEP 6: Generating Physiological Pathway Insights")
        print("="*80)
        
        insights = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'clinical_records': len(self.clinical_data),
                'socioeconomic_records': len(self.socio_data),
                'integrated_records': len(self.integrated_data),
                'balanced_ratio': len(self.clinical_data) / len(self.integrated_data) if len(self.integrated_data) > 0 else 0
            },
            'biomarker_pathways': {},
            'key_findings': [],
            'physiological_interpretations': []
        }
        
        # Analyze each biomarker pathway
        for biomarker, results in self.pathway_results.items():
            pathway_imp = results['pathway_importance']
            
            # Determine dominant pathway
            dominant_pathway = max(pathway_imp, key=pathway_imp.get)
            
            insights['biomarker_pathways'][biomarker] = {
                'dominant_pathway': dominant_pathway,
                'pathway_contributions': pathway_imp,
                'model_r2': results['model_performance']['r2'],
                'sample_size': results['n_samples'],
                'top_features': sorted(results['feature_importance'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
            }
            
            # Generate physiological interpretation
            if biomarker == 'CD4 cell count (cells/µL)' and dominant_pathway == 'demographic':
                insights['physiological_interpretations'].append(
                    f"CD4 count shows strong demographic dependency (age, sex), suggesting "
                    f"baseline immunological variation across population groups rather than "
                    f"environmental modulation."
                )
            elif biomarker == 'Hemoglobin (g/dL)' and pathway_imp.get('climate', 0) > 20:
                insights['physiological_interpretations'].append(
                    f"Hemoglobin levels show {pathway_imp['climate']:.1f}% climate contribution, "
                    f"potentially reflecting heat-induced plasma volume changes and dehydration effects."
                )
            elif 'blood pressure' in biomarker and pathway_imp.get('socioeconomic', 0) > 30:
                insights['physiological_interpretations'].append(
                    f"{biomarker} shows {pathway_imp['socioeconomic']:.1f}% socioeconomic contribution, "
                    f"indicating lifestyle and stress-related factors influence cardiovascular regulation."
                )
        
        # Generate key findings
        if len(self.pathway_results) > 0:
            # Find biomarkers most influenced by climate
            climate_sensitive = [(bio, res['pathway_importance'].get('climate', 0)) 
                               for bio, res in self.pathway_results.items()]
            climate_sensitive.sort(key=lambda x: x[1], reverse=True)
            
            if climate_sensitive[0][1] > 20:
                insights['key_findings'].append(
                    f"{climate_sensitive[0][0]} shows highest climate sensitivity "
                    f"({climate_sensitive[0][1]:.1f}% of variation explained by temperature variables)"
                )
            
            # Find biomarkers with best model performance
            best_models = [(bio, res['model_performance']['r2']) 
                          for bio, res in self.pathway_results.items()]
            best_models.sort(key=lambda x: x[1], reverse=True)
            
            if best_models[0][1] > 0.1:
                insights['key_findings'].append(
                    f"Best predictive model achieved for {best_models[0][0]} "
                    f"(R² = {best_models[0][1]:.3f}), indicating measurable environmental-physiological relationships"
                )
            
            # Overall pathway dominance
            avg_pathways = {'demographic': [], 'socioeconomic': [], 'climate': []}
            for res in self.pathway_results.values():
                for pathway, value in res['pathway_importance'].items():
                    avg_pathways[pathway].append(value)
            
            for pathway, values in avg_pathways.items():
                if values:
                    avg_pathways[pathway] = np.mean(values)
            
            dominant_overall = max(avg_pathways, key=avg_pathways.get)
            insights['key_findings'].append(
                f"Across all biomarkers, {dominant_overall} factors show strongest influence "
                f"({avg_pathways[dominant_overall]:.1f}% average contribution)"
            )
        
        # Save insights
        insights_path = self.results_path / "physiological_pathway_insights.json"
        with open(insights_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        print("\nKey Physiological Insights:")
        for finding in insights['key_findings']:
            print(f"  • {finding}")
        
        print("\nPhysiological Interpretations:")
        for interp in insights['physiological_interpretations']:
            print(f"  • {interp}")
        
        return insights
    
    def create_integrated_visualization(self):
        """Create comprehensive visualization of pathway analysis."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Dataset composition
        ax1 = fig.add_subplot(gs[0, 0])
        if 'data_source' in self.integrated_data.columns:
            source_counts = self.integrated_data['data_source'].value_counts()
            ax1.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                   colors=['#FF6B6B', '#4ECDC4'])
            ax1.set_title('Integrated Dataset Composition')
        
        # 2. Biomarker coverage
        ax2 = fig.add_subplot(gs[0, 1])
        biomarker_coverage = []
        biomarker_names = []
        for bio in self.pathway_results.keys():
            biomarker_coverage.append(self.pathway_results[bio]['n_samples'])
            biomarker_names.append(bio.split('(')[0][:15])  # Truncate names
        
        if biomarker_coverage:
            bars = ax2.bar(range(len(biomarker_coverage)), biomarker_coverage, color='#45B7D1')
            ax2.set_xticks(range(len(biomarker_names)))
            ax2.set_xticklabels(biomarker_names, rotation=45, ha='right')
            ax2.set_ylabel('Sample Size')
            ax2.set_title('Biomarker Data Availability')
        
        # 3. Model performance comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if self.pathway_results:
            r2_scores = [res['model_performance']['r2'] for res in self.pathway_results.values()]
            biomarkers_short = [bio.split('(')[0][:15] for bio in self.pathway_results.keys()]
            
            bars = ax3.barh(range(len(r2_scores)), r2_scores, color='#95E77E')
            ax3.set_yticks(range(len(biomarkers_short)))
            ax3.set_yticklabels(biomarkers_short)
            ax3.set_xlabel('R² Score')
            ax3.set_title('Model Performance by Biomarker')
            ax3.set_xlim(-0.1, 1.0)
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4-6. Pathway importance for each biomarker
        for idx, (biomarker, results) in enumerate(list(self.pathway_results.items())[:3]):
            ax = fig.add_subplot(gs[1, idx])
            pathway_imp = results['pathway_importance']
            
            colors = {'demographic': '#FF6B6B', 'socioeconomic': '#4ECDC4', 'climate': '#45B7D1'}
            bar_colors = [colors[p] for p in pathway_imp.keys()]
            
            bars = ax.bar(pathway_imp.keys(), pathway_imp.values(), color=bar_colors)
            ax.set_ylabel('Importance (%)')
            ax.set_title(f'{biomarker.split("(")[0][:20]}\nPathway Contributions')
            ax.set_ylim(0, 100)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
        
        # 7. Average pathway importance across all biomarkers
        ax7 = fig.add_subplot(gs[2, 0])
        if self.pathway_results:
            avg_pathways = {'demographic': [], 'socioeconomic': [], 'climate': []}
            for res in self.pathway_results.values():
                for pathway, value in res['pathway_importance'].items():
                    avg_pathways[pathway].append(value)
            
            avg_values = {k: np.mean(v) if v else 0 for k, v in avg_pathways.items()}
            
            colors = {'demographic': '#FF6B6B', 'socioeconomic': '#4ECDC4', 'climate': '#45B7D1'}
            bar_colors = [colors[p] for p in avg_values.keys()]
            
            bars = ax7.bar(avg_values.keys(), avg_values.values(), color=bar_colors)
            ax7.set_ylabel('Average Importance (%)')
            ax7.set_title('Overall Pathway Contributions\n(Averaged Across Biomarkers)')
            ax7.set_ylim(0, 100)
            
            for bar in bars:
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        # 8. Feature importance heatmap
        ax8 = fig.add_subplot(gs[2, 1:])
        if self.pathway_results:
            # Create feature importance matrix
            all_features = set()
            for res in self.pathway_results.values():
                all_features.update(res['feature_importance'].keys())
            
            all_features = sorted(list(all_features))
            biomarker_names = list(self.pathway_results.keys())
            
            importance_matrix = np.zeros((len(biomarker_names), len(all_features)))
            for i, bio in enumerate(biomarker_names):
                for j, feat in enumerate(all_features):
                    importance_matrix[i, j] = self.pathway_results[bio]['feature_importance'].get(feat, 0)
            
            # Normalize by row (biomarker)
            row_sums = importance_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            importance_matrix = importance_matrix / row_sums * 100
            
            im = ax8.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
            ax8.set_xticks(range(len(all_features)))
            ax8.set_xticklabels(all_features, rotation=45, ha='right')
            ax8.set_yticks(range(len(biomarker_names)))
            ax8.set_yticklabels([b.split('(')[0][:20] for b in biomarker_names])
            ax8.set_title('Feature Importance Heatmap (%)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax8)
            cbar.set_label('Relative Importance (%)')
        
        # Add main title
        fig.suptitle('Optimized Biomarker Pathway Analysis\nIntegrating Clinical, Socioeconomic, and Climate Data', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        fig_path = self.figures_path / "integrated_pathway_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved integrated visualization: {fig_path}")
    
    def run_complete_analysis(self):
        """Run the complete optimized analysis pipeline."""
        print("\n" + "="*80)
        print("OPTIMIZED BIOMARKER PATHWAY ANALYSIS PIPELINE")
        print("Focusing on physiological pathways in environmental context")
        print("="*80)
        
        # Step 1: Analyze clinical geography
        self.load_and_analyze_clinical_geography()
        
        # Step 2: Load and filter socioeconomic data
        self.load_and_filter_socioeconomic_data()
        
        # Step 3: Load climate data
        self.load_climate_data()
        
        # Step 4: Create balanced integrated dataset
        self.create_balanced_integrated_dataset()
        
        # Step 5: Build biomarker pathway models
        self.build_biomarker_pathway_models()
        
        # Step 6: Generate physiological insights
        insights = self.generate_physiological_insights()
        
        # Create integrated visualization
        self.create_integrated_visualization()
        
        # Save comprehensive results
        comprehensive_results = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'pipeline': 'Optimized Biomarker Pathway Analysis',
                'focus': 'Physiological pathways in climate-socioeconomic context'
            },
            'dataset_composition': {
                'clinical_records': len(self.clinical_data),
                'socioeconomic_records': len(self.socio_data),
                'climate_records': len(self.climate_data),
                'integrated_records': len(self.integrated_data)
            },
            'clinical_geography': self.clinical_geography,
            'pathway_models': {
                bio: {
                    'performance': res['model_performance'],
                    'pathway_importance': res['pathway_importance'],
                    'top_features': sorted(res['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                }
                for bio, res in self.pathway_results.items()
            },
            'insights': insights
        }
        
        results_path = self.results_path / "optimized_pathway_analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.results_path}")
        print(f"Visualizations saved to: {self.figures_path}")
        print(f"\nKey outputs:")
        print(f"  • Balanced integrated dataset: balanced_integrated_dataset.csv")
        print(f"  • Pathway analysis results: optimized_pathway_analysis_results.json")
        print(f"  • Physiological insights: physiological_pathway_insights.json")
        print(f"  • Visualizations: {len(list(self.figures_path.glob('*.png')))} figures generated")
        
        return comprehensive_results


if __name__ == "__main__":
    # Run the optimized analysis
    analyzer = OptimizedBiomarkerPathwayAnalysis()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("Optimized analysis complete!")
    print("This pipeline balances clinical and socioeconomic data")
    print("to better understand physiological pathways.")
    print("="*80)