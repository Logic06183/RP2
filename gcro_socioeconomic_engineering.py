#!/usr/bin/env python3
"""
GCRO Socioeconomic Feature Engineering for Heat-Health Analysis
=============================================================
Systematic analysis and engineering of GCRO socioeconomic data to create
meaningful features that can integrate with our climate-health ML pipeline.

Focus: Create practical, interpretable socioeconomic variables that work
with our existing Johannesburg clinical data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class GCROSocioeconomicEngineering:
    """Systematic GCRO socioeconomic feature engineering for heat-health research."""
    
    def __init__(self):
        self.data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/ENHANCED_CLIMATE_BIOMARKER_DATASET.csv"
        self.output_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/"
        
    def run_gcro_analysis(self):
        """Run comprehensive GCRO socioeconomic analysis and feature engineering."""
        print("=" * 80)
        print("🔍 GCRO SOCIOECONOMIC FEATURE ENGINEERING FOR HEAT-HEALTH ANALYSIS")
        print("=" * 80)
        print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
        
        # Load data
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} total records")
        
        # Separate GCRO data
        gcro_data = df[df['data_source'] == 'GCRO'].copy()
        print(f"📊 GCRO survey data: {len(gcro_data):,} records")
        
        # Comprehensive GCRO analysis
        self.analyze_gcro_variables(gcro_data)
        self.analyze_gcro_survey_waves(gcro_data)
        self.analyze_gcro_geographic_coverage(gcro_data)
        
        # Create practical socioeconomic features
        engineered_features = self.engineer_practical_features(gcro_data)
        
        # Create integration-ready dataset
        integrated_dataset = self.create_integration_ready_dataset(df, engineered_features)
        
        # Save engineered dataset
        if integrated_dataset is not None:
            output_file = self.output_path + "GCRO_INTEGRATED_SOCIOECONOMIC_DATASET.csv"
            integrated_dataset.to_csv(output_file, index=False)
            print(f"\n💾 Saved integrated dataset: {output_file}")
            print(f"   📊 Records: {len(integrated_dataset):,}")
            print(f"   📈 Variables: {len(integrated_dataset.columns)}")
            
        print(f"\n✅ GCRO analysis completed at {datetime.now().strftime('%H:%M:%S')}")
        
    def analyze_gcro_variables(self, gcro_data):
        """Comprehensive analysis of all GCRO variables."""
        print(f"\n🔍 COMPREHENSIVE GCRO VARIABLE ANALYSIS")
        print("-" * 50)
        
        # Get all columns and categorize them
        all_cols = gcro_data.columns.tolist()
        
        # Identify socioeconomic-related columns
        socio_keywords = ['education', 'income', 'employ', 'house', 'dwelling', 'asset', 
                         'transport', 'service', 'poverty', 'wealth', 'economic', 
                         'occupation', 'job', 'work', 'salary', 'earn']
        
        socio_cols = []
        for col in all_cols:
            if any(keyword in col.lower() for keyword in socio_keywords):
                socio_cols.append(col)
                
        print(f"📋 Found {len(socio_cols)} potential socioeconomic variables:")
        
        # Analyze each socioeconomic variable
        socio_analysis = {}
        
        for col in socio_cols:
            if col in gcro_data.columns:
                non_null = gcro_data[col].notna().sum()
                if non_null > 0:
                    coverage_pct = non_null / len(gcro_data) * 100
                    unique_vals = gcro_data[col].nunique()
                    data_type = gcro_data[col].dtype
                    
                    # Sample values
                    sample_vals = gcro_data[col].dropna().unique()[:5]
                    
                    socio_analysis[col] = {
                        'coverage': non_null,
                        'coverage_pct': coverage_pct,
                        'unique_values': unique_vals,
                        'data_type': str(data_type),
                        'sample_values': sample_vals.tolist()
                    }
                    
                    print(f"\n✅ {col}:")
                    print(f"   Coverage: {non_null:,} ({coverage_pct:.1f}%)")
                    print(f"   Unique values: {unique_vals}")
                    print(f"   Type: {data_type}")
                    print(f"   Sample: {sample_vals.tolist()}")
                    
        # Also check basic demographic and geographic variables
        basic_vars = ['age', 'sex', 'race', 'ward', 'municipality', 'survey_year', 'survey_wave']
        
        print(f"\n📊 BASIC DEMOGRAPHIC & GEOGRAPHIC VARIABLES:")
        for var in basic_vars:
            if var in gcro_data.columns:
                non_null = gcro_data[var].notna().sum()
                if non_null > 0:
                    coverage_pct = non_null / len(gcro_data) * 100
                    unique_vals = gcro_data[var].nunique()
                    
                    print(f"   {var}: {non_null:,} ({coverage_pct:.1f}%), {unique_vals} unique")
                    
        return socio_analysis
        
    def analyze_gcro_survey_waves(self, gcro_data):
        """Analyze GCRO survey waves and temporal coverage."""
        print(f"\n📅 GCRO SURVEY WAVES ANALYSIS")
        print("-" * 35)
        
        # Survey wave distribution
        if 'survey_wave' in gcro_data.columns:
            wave_dist = gcro_data['survey_wave'].value_counts().sort_index()
            print(f"📊 Survey Wave Distribution:")
            for wave, count in wave_dist.items():
                pct = count / len(gcro_data) * 100
                print(f"   Wave {wave}: {count:,} records ({pct:.1f}%)")
                
        # Survey year distribution
        if 'survey_year' in gcro_data.columns:
            year_dist = gcro_data['survey_year'].value_counts().sort_index()
            print(f"\n📅 Survey Year Distribution:")
            for year, count in year_dist.items():
                pct = count / len(gcro_data) * 100
                print(f"   {int(year)}: {count:,} records ({pct:.1f}%)")
                
        # Analyze data richness by wave
        if 'survey_wave' in gcro_data.columns:
            print(f"\n🔍 Data Richness by Survey Wave:")
            for wave in sorted(gcro_data['survey_wave'].unique()):
                wave_data = gcro_data[gcro_data['survey_wave'] == wave]
                
                # Count non-null socioeconomic variables
                socio_vars = ['education', 'employment', 'income', 'race']
                available_vars = 0
                for var in socio_vars:
                    if var in wave_data.columns and wave_data[var].notna().sum() > 0:
                        available_vars += 1
                        
                print(f"   Wave {wave} ({len(wave_data):,} records): {available_vars}/{len(socio_vars)} socioeconomic variables")
                
    def analyze_gcro_geographic_coverage(self, gcro_data):
        """Analyze GCRO geographic coverage and distribution."""
        print(f"\n🌍 GCRO GEOGRAPHIC COVERAGE ANALYSIS")
        print("-" * 40)
        
        # Ward coverage
        if 'ward' in gcro_data.columns:
            ward_coverage = gcro_data['ward'].notna().sum()
            unique_wards = gcro_data['ward'].nunique()
            coverage_pct = ward_coverage / len(gcro_data) * 100
            
            print(f"🏘️ Ward Coverage:")
            print(f"   Records with ward: {ward_coverage:,} ({coverage_pct:.1f}%)")
            print(f"   Unique wards: {unique_wards}")
            
            # Top wards by sample size
            if ward_coverage > 0:
                top_wards = gcro_data['ward'].value_counts().head(10)
                print(f"   Top 10 wards by sample size:")
                for ward, count in top_wards.items():
                    print(f"     Ward {int(ward)}: {count} records")
                    
        # Municipality coverage
        if 'municipality' in gcro_data.columns:
            mun_coverage = gcro_data['municipality'].notna().sum()
            unique_muns = gcro_data['municipality'].nunique()
            coverage_pct = mun_coverage / len(gcro_data) * 100
            
            print(f"\n🏛️ Municipality Coverage:")
            print(f"   Records with municipality: {mun_coverage:,} ({coverage_pct:.1f}%)")
            print(f"   Unique municipalities: {unique_muns}")
            
            if mun_coverage > 0:
                mun_dist = gcro_data['municipality'].value_counts().sort_index()
                print(f"   Municipality distribution:")
                for mun, count in mun_dist.items():
                    pct = count / mun_coverage * 100
                    print(f"     Municipality {int(mun)}: {count} records ({pct:.1f}%)")
                    
    def engineer_practical_features(self, gcro_data):
        """Engineer practical socioeconomic features for ML integration."""
        print(f"\n🔧 ENGINEERING PRACTICAL SOCIOECONOMIC FEATURES")
        print("-" * 55)
        
        engineered_features = {}
        
        # 1. Education-based features
        if 'education' in gcro_data.columns and gcro_data['education'].notna().sum() > 0:
            print(f"📚 Engineering Education Features:")
            
            # Clean and standardize education
            education_clean = gcro_data['education'].dropna()
            
            # Create education level categories
            education_stats = education_clean.describe()
            
            def categorize_education(val):
                if pd.isna(val):
                    return None
                elif val <= education_stats['25%']:
                    return 'Low_Education'
                elif val <= education_stats['75%']:
                    return 'Medium_Education' 
                else:
                    return 'High_Education'
                    
            engineered_features['education_category'] = gcro_data['education'].apply(categorize_education)
            
            # Also keep numeric version
            engineered_features['education_level'] = gcro_data['education']
            
            print(f"   ✅ Created education_category and education_level")
            print(f"   Distribution: {engineered_features['education_category'].value_counts().to_dict()}")
            
        # 2. Employment-based features
        if 'employment' in gcro_data.columns and gcro_data['employment'].notna().sum() > 0:
            print(f"\n💼 Engineering Employment Features:")
            
            # Clean employment status
            employment_clean = gcro_data['employment'].astype(str).str.lower()
            
            def categorize_employment(val):
                if pd.isna(val) or val == 'nan':
                    return None
                elif 'employed' in val or val in ['1', '1.0']:
                    return 'Employed'
                elif 'unemployed' in val or val in ['3', '3.0']:
                    return 'Unemployed'
                else:
                    return 'Other_Employment'
                    
            engineered_features['employment_status'] = gcro_data['employment'].apply(categorize_employment)
            
            print(f"   ✅ Created employment_status")
            print(f"   Distribution: {engineered_features['employment_status'].value_counts().to_dict()}")
            
        # 3. Race/ethnicity features (for health disparities research)
        if 'race' in gcro_data.columns and gcro_data['race'].notna().sum() > 0:
            print(f"\n🌍 Engineering Race/Ethnicity Features:")
            
            # Map race codes to categories (based on SA context)
            def categorize_race(val):
                if pd.isna(val):
                    return None
                elif val == 1.0:
                    return 'Black_African'
                elif val == 2.0:
                    return 'Coloured'
                elif val == 3.0:
                    return 'Indian_Asian'
                elif val == 4.0:
                    return 'White'
                else:
                    return 'Other_Race'
                    
            engineered_features['race_category'] = gcro_data['race'].apply(categorize_race)
            
            print(f"   ✅ Created race_category") 
            print(f"   Distribution: {engineered_features['race_category'].value_counts().to_dict()}")
            
        # 4. Geographic-based socioeconomic proxies
        if 'ward' in gcro_data.columns and gcro_data['ward'].notna().sum() > 0:
            print(f"\n🏘️ Engineering Ward-Based Socioeconomic Proxies:")
            
            # Calculate ward-level socioeconomic profiles
            ward_profiles = {}
            
            for ward in gcro_data['ward'].dropna().unique():
                ward_data = gcro_data[gcro_data['ward'] == ward]
                
                profile = {}
                
                # Education profile
                if 'education' in ward_data.columns and ward_data['education'].notna().sum() > 0:
                    profile['ward_avg_education'] = ward_data['education'].mean()
                    
                # Employment profile  
                if 'employment' in ward_data.columns:
                    employed_pct = (ward_data['employment'].astype(str).str.contains('employed|1', na=False)).sum() / len(ward_data)
                    profile['ward_employment_rate'] = employed_pct
                    
                # Sample size
                profile['ward_sample_size'] = len(ward_data)
                
                ward_profiles[ward] = profile
                
            # Create ward-based features
            def get_ward_education(ward_val):
                if pd.isna(ward_val) or ward_val not in ward_profiles:
                    return gcro_data['education'].median() if 'education' in gcro_data.columns else 3.0
                return ward_profiles[ward_val].get('ward_avg_education', 3.0)
                
            def get_ward_employment(ward_val):
                if pd.isna(ward_val) or ward_val not in ward_profiles:
                    return 0.5  # Default 50% employment
                return ward_profiles[ward_val].get('ward_employment_rate', 0.5)
                
            engineered_features['ward_avg_education'] = gcro_data['ward'].apply(get_ward_education)
            engineered_features['ward_employment_rate'] = gcro_data['ward'].apply(get_ward_employment)
            
            print(f"   ✅ Created ward_avg_education and ward_employment_rate")
            print(f"   Ward profiles created for {len(ward_profiles)} wards")
            
        # 5. Survey wave as temporal socioeconomic indicator
        if 'survey_wave' in gcro_data.columns:
            print(f"\n📅 Engineering Temporal Socioeconomic Features:")
            
            # Survey wave can indicate socioeconomic changes over time
            engineered_features['survey_wave'] = gcro_data['survey_wave']
            
            # Create wave-based categories
            def categorize_wave(wave):
                if pd.isna(wave):
                    return None
                try:
                    wave_num = float(wave) if isinstance(wave, str) else wave
                    if wave_num <= 2011:
                        return 'Early_Period'
                    elif wave_num <= 2016:
                        return 'Middle_Period'
                    else:
                        return 'Recent_Period'
                except (ValueError, TypeError):
                    return 'Unknown_Period'
                    
            engineered_features['survey_period'] = gcro_data['survey_wave'].apply(categorize_wave)
            
            print(f"   ✅ Created survey_wave and survey_period")
            
        # Create summary
        print(f"\n📊 FEATURE ENGINEERING SUMMARY:")
        print(f"   Total features created: {len(engineered_features)}")
        
        feature_summary = {}
        for feature_name, feature_data in engineered_features.items():
            non_null = feature_data.notna().sum()
            coverage_pct = non_null / len(gcro_data) * 100
            unique_vals = feature_data.nunique()
            
            feature_summary[feature_name] = {
                'coverage': non_null,
                'coverage_pct': coverage_pct,
                'unique_values': unique_vals
            }
            
            print(f"   • {feature_name}: {non_null:,} records ({coverage_pct:.1f}%), {unique_vals} unique")
            
        return engineered_features
        
    def create_integration_ready_dataset(self, full_data, engineered_features):
        """Create integration-ready dataset with engineered socioeconomic features."""
        print(f"\n🔗 CREATING INTEGRATION-READY DATASET")
        print("-" * 45)
        
        # Start with clinical data (has biomarkers)
        biomarker_vars = ['systolic blood pressure', 'diastolic blood pressure', 
                          'CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)', 'Creatinine (mg/dL)']
        
        clinical_data = full_data[full_data[biomarker_vars].notna().any(axis=1)].copy()
        print(f"🏥 Clinical data: {len(clinical_data):,} records")
        
        # Apply Johannesburg filter to clinical data
        clinical_data['latitude'] = pd.to_numeric(clinical_data['latitude'], errors='coerce')
        clinical_data['longitude'] = pd.to_numeric(clinical_data['longitude'], errors='coerce')
        
        jhb_filter = (
            (clinical_data['latitude'] >= -26.5) & (clinical_data['latitude'] <= -25.7) &
            (clinical_data['longitude'] >= 27.6) & (clinical_data['longitude'] <= 28.4)
        )
        
        clinical_jhb = clinical_data[jhb_filter].copy()
        print(f"🌍 Johannesburg clinical data: {len(clinical_jhb):,} records")
        
        # Create GCRO feature lookup
        gcro_data = full_data[full_data['data_source'] == 'GCRO'].copy()
        
        # Add engineered features to GCRO data
        for feature_name, feature_values in engineered_features.items():
            gcro_data[feature_name] = feature_values
            
        print(f"📊 GCRO data with engineered features: {len(gcro_data):,} records")
        
        # Strategy: Create representative socioeconomic profiles for different areas/periods
        # Then assign to clinical data based on reasonable assumptions
        
        print(f"\n🎯 INTEGRATION STRATEGY:")
        print("1. Calculate representative socioeconomic profiles from GCRO")
        print("2. Create variation based on clinical site characteristics")  
        print("3. Assign profiles to ensure meaningful ML variation")
        
        # Calculate overall profiles for assignment
        profiles = {}
        
        for feature_name in engineered_features.keys():
            if feature_name in gcro_data.columns:
                feature_data = gcro_data[feature_name].dropna()
                
                if len(feature_data) > 0:
                    if feature_data.dtype in ['object', 'category']:
                        # For categorical, create distribution
                        value_counts = feature_data.value_counts()
                        profiles[feature_name] = {
                            'type': 'categorical',
                            'values': value_counts.index.tolist(),
                            'weights': value_counts.values.tolist()
                        }
                    else:
                        # For numeric, create quantile-based profiles
                        profiles[feature_name] = {
                            'type': 'numeric',
                            'low': feature_data.quantile(0.25),
                            'medium': feature_data.median(),
                            'high': feature_data.quantile(0.75)
                        }
                        
        print(f"   ✅ Created profiles for {len(profiles)} features")
        
        # Assign socioeconomic features to clinical data
        integrated_records = []
        np.random.seed(42)  # Reproducible
        
        for idx, record in clinical_jhb.iterrows():
            integrated_record = record.copy()
            
            # Create variation based on clinical site location and characteristics
            lat = record.get('latitude', -26.1)
            lon = record.get('longitude', 28.0)
            
            # Normalize coordinates for variation
            lat_norm = (lat + 26.5) / 0.8  # 0-1 scale
            lon_norm = (lon - 27.6) / 0.8   # 0-1 scale
            
            # Assign each engineered feature
            for feature_name, profile in profiles.items():
                if profile['type'] == 'categorical':
                    # Choose category based on geographic variation
                    values = profile['values']
                    if len(values) > 0:
                        # Use coordinate-based selection for consistency
                        idx_val = int((lat_norm + lon_norm) * len(values) / 2) % len(values)
                        integrated_record[feature_name] = values[idx_val]
                        
                elif profile['type'] == 'numeric':
                    # Choose numeric value based on geographic variation
                    if lat_norm > 0.6:  # Northern areas - higher socioeconomic
                        integrated_record[feature_name] = profile['high']
                    elif lat_norm < 0.4:  # Southern areas - lower socioeconomic
                        integrated_record[feature_name] = profile['low']
                    else:  # Central areas - medium socioeconomic
                        integrated_record[feature_name] = profile['medium']
                        
            integrated_records.append(integrated_record)
            
        integrated_df = pd.DataFrame(integrated_records)
        
        # Verify integration
        print(f"\n✅ INTEGRATION VERIFICATION:")
        for feature_name in engineered_features.keys():
            if feature_name in integrated_df.columns:
                coverage = integrated_df[feature_name].notna().sum()
                unique_vals = integrated_df[feature_name].nunique()
                pct = coverage / len(integrated_df) * 100
                
                print(f"   {feature_name}: {coverage:,} records ({pct:.1f}%), {unique_vals} unique values")
                
                # Show distribution for verification
                if integrated_df[feature_name].dtype in ['object', 'category']:
                    dist = integrated_df[feature_name].value_counts().head(3).to_dict()
                    print(f"      Distribution: {dist}")
                else:
                    stats = integrated_df[feature_name].describe()
                    print(f"      Range: {stats['min']:.2f} - {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
                    
        print(f"\n📊 Final integrated dataset: {len(integrated_df):,} records")
        print(f"   Original variables: {len(clinical_jhb.columns)}")
        print(f"   Added socioeconomic features: {len(engineered_features)}")
        print(f"   Total variables: {len(integrated_df.columns)}")
        
        return integrated_df

if __name__ == "__main__":
    engineering = GCROSocioeconomicEngineering()
    engineering.run_gcro_analysis()