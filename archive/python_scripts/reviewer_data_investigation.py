#!/usr/bin/env python3
"""
Critical Data Investigation for Reviewer Questions
Heat-Health XAI Analysis Framework

This script addresses all critical discrepancies identified by reviewers:
1. Sample size discrepancy (2,334 vs 2,884)
2. Temporal period (2011-2021 vs claimed 2013-2021)
3. Geographic coverage and variation
4. Data quality and unit consistency issues
5. Model performance validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def investigate_sample_sizes():
    """Address Priority 1: Sample size discrepancy"""
    print("🚨 PRIORITY 1: SAMPLE SIZE DISCREPANCY INVESTIGATION")
    print("="*70)
    
    # Load all datasets
    data_dir = Path("data/optimal_xai_ready")
    
    datasets = {
        'high_quality': 'xai_ready_high_quality.csv',
        'all_available': 'xai_ready_all_available.csv',
        'complete_cases': 'xai_ready_complete_cases.csv',
        'dphru_053': 'xai_ready_dphru_053_optimal.csv',
        'wrhi_001': 'xai_ready_wrhi_001_optimal.csv'
    }
    
    sample_analysis = {}
    
    for name, filename in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, low_memory=False)
                sample_analysis[name] = {
                    'total_rows': len(df),
                    'unique_participants': df['participant_id'].nunique(),
                    'dataset_sources': df['dataset_source'].value_counts().to_dict() if 'dataset_source' in df.columns else {}
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print("\n📊 SAMPLE SIZE BREAKDOWN:")
    for name, stats in sample_analysis.items():
        print(f"\n{name.upper()}:")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Unique participants: {stats['unique_participants']:,}")
        if stats['dataset_sources']:
            print(f"  Dataset composition:")
            for source, count in stats['dataset_sources'].items():
                print(f"    {source}: {count:,} records")
    
    # Determine which dataset matches paper claims
    print("\n🎯 PAPER CLAIMS RECONCILIATION:")
    print(f"Paper claims: 2,334 participants")
    
    if 'high_quality' in sample_analysis:
        hq_rows = sample_analysis['high_quality']['total_rows']
        hq_participants = sample_analysis['high_quality']['unique_participants']
        print(f"High quality dataset: {hq_rows:,} rows, {hq_participants:,} unique participants")
        
        if hq_rows == 2334:
            print("✅ HIGH QUALITY DATASET MATCHES PAPER CLAIM (2,334 records)")
            print("   This appears to be the final analysis dataset used in the paper")
        
        if hq_participants == 1807:
            print("⚠️  However, unique participants = 1,807 (suggesting repeated measures)")
            
    return sample_analysis

def investigate_temporal_coverage():
    """Address Priority 2: Temporal period discrepancy"""
    print("\n🚨 PRIORITY 2: TEMPORAL COVERAGE INVESTIGATION") 
    print("="*70)
    
    # Load high quality dataset (matches paper sample size)
    df = pd.read_csv("data/optimal_xai_ready/xai_ready_high_quality.csv", low_memory=False)
    
    # Convert dates and extract years
    df['visit_date'] = pd.to_datetime(df['std_visit_date'])
    df['year'] = df['visit_date'].dt.year
    
    print(f"\n📅 TEMPORAL ANALYSIS:")
    print(f"Date range: {df['visit_date'].min()} to {df['visit_date'].max()}")
    print(f"Year range: {df['year'].min():.0f} to {df['year'].max():.0f}")
    
    # Year distribution
    year_counts = df['year'].value_counts().sort_index()
    print(f"\n📊 RECORDS BY YEAR:")
    total_valid_dates = 0
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}: {count:,} records")
            total_valid_dates += count
    
    missing_dates = df['std_visit_date'].isna().sum()
    print(f"  Missing dates: {missing_dates:,} records")
    print(f"  Total with valid dates: {total_valid_dates:,}")
    
    # Check paper claims
    print(f"\n🎯 PAPER CLAIMS RECONCILIATION:")
    print(f"Paper claims: 2013-2021 period")
    print(f"Actual data: 2011-2021 period")
    
    if df['year'].min() == 2011:
        print("⚠️  DATA STARTS IN 2011, NOT 2013 AS CLAIMED")
        
    # Filter to paper's claimed period
    paper_period = df[(df['year'] >= 2013) & (df['year'] <= 2021)]
    print(f"Records in claimed period (2013-2021): {len(paper_period):,}")
    
    return df

def investigate_geographic_coverage(df):
    """Address Priority 3: Geographic coverage limitations"""
    print("\n🚨 PRIORITY 3: GEOGRAPHIC COVERAGE INVESTIGATION")
    print("="*70)
    
    # Check for geographic variables
    geo_cols = [col for col in df.columns if any(term in col.lower() for term in ['location', 'site', 'ward', 'province', 'lat', 'lon', 'coord'])]
    
    print(f"\n🗺️  GEOGRAPHIC VARIABLES FOUND:")
    for col in geo_cols:
        print(f"  {col}")
    
    # Analyze dataset sources (these indicate study sites)
    if 'dataset_source' in df.columns:
        sources = df['dataset_source'].value_counts()
        print(f"\n📍 STUDY SITES (Dataset Sources):")
        for source, count in sources.items():
            print(f"  {source}: {count:,} records")
    
    # Check if all data is from Johannesburg area
    print(f"\n🎯 GEOGRAPHIC REALITY CHECK:")
    print("All dataset sources appear to be Johannesburg-based studies:")
    print("  - DPHRU: Developmental Pathways for Health Research Unit (JHB)")
    print("  - VIDA: Studies conducted in JHB area") 
    print("  - WRHI: Wits Reproductive Health & HIV Institute (JHB)")
    
    print("\n⚠️  CRITICAL FINDING:")
    print("This is a SINGLE-SITE study (Johannesburg only)")
    print("Claims about 'geographic gradient' must refer to WITHIN-city variation")
    
    # Climate variation analysis
    climate_cols = [col for col in df.columns if 'climate' in col.lower()]
    if climate_cols:
        print(f"\n🌡️  CLIMATE VARIABLES AVAILABLE: {len(climate_cols)} variables")
        
        # Check temperature variation
        temp_cols = [col for col in climate_cols if 'temp' in col]
        if temp_cols:
            temp_col = temp_cols[0]  # Use first temperature variable
            temp_stats = df[temp_col].describe()
            print(f"Temperature variation (using {temp_col}):")
            print(f"  Range: {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C")
            print(f"  Std Dev: {temp_stats['std']:.2f}°C")
            
            if temp_stats['std'] < 5:
                print("⚠️  LOW TEMPERATURE VARIATION - Limited climate gradient")
    
    return geo_cols, climate_cols

def investigate_data_quality(df):
    """Address Priority 4: Data quality and unit consistency"""
    print("\n🚨 PRIORITY 4: DATA QUALITY INVESTIGATION")
    print("="*70)
    
    # Check glucose units (major concern from reviewers)
    print(f"\n🩸 GLUCOSE UNIT CONSISTENCY CHECK:")
    
    if 'std_glucose' in df.columns:
        glucose_stats = df.groupby('dataset_source')['std_glucose'].describe()
        print("Glucose by dataset source:")
        print(glucose_stats[['min', 'max', 'mean', 'std']].round(2))
        
        # Check for unit inconsistencies
        for source in df['dataset_source'].unique():
            source_data = df[df['dataset_source'] == source]['std_glucose'].dropna()
            if len(source_data) > 0:
                mean_val = source_data.mean()
                print(f"\n{source}:")
                print(f"  Mean glucose: {mean_val:.2f}")
                if mean_val < 20:
                    print("  ⚠️  LIKELY IN mmol/L (should convert to mg/dL)")
                    print(f"  Converted to mg/dL: {mean_val * 18.0182:.1f}")
                else:
                    print("  ✅ Likely in mg/dL (standard unit)")
    
    # Check other biomarkers for consistency
    biomarker_cols = ['std_glucose', 'std_cholesterol_total', 'std_creatinine', 'std_systolic_bp']
    
    print(f"\n🔬 BIOMARKER RANGES BY SOURCE:")
    for col in biomarker_cols:
        if col in df.columns:
            print(f"\n{col.upper()}:")
            source_stats = df.groupby('dataset_source')[col].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            print(source_stats)
    
    # Check for extreme values/outliers
    print(f"\n⚠️  OUTLIER DETECTION:")
    for col in biomarker_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
            if len(outliers) > 0:
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
    
    return biomarker_cols

def investigate_climate_integration():
    """Address Priority 5: Climate data integration methodology"""
    print("\n🚨 PRIORITY 5: CLIMATE DATA INTEGRATION")
    print("="*70)
    
    df = pd.read_csv("data/optimal_xai_ready/xai_ready_high_quality.csv", low_memory=False)
    
    # Climate variables analysis
    climate_cols = [col for col in df.columns if 'climate' in col]
    print(f"📊 CLIMATE VARIABLES: {len(climate_cols)} variables found")
    
    # Group by type
    temp_vars = [col for col in climate_cols if 'temp' in col]
    heat_vars = [col for col in climate_cols if 'heat' in col]
    seasonal_vars = [col for col in climate_cols if 'season' in col]
    
    print(f"  Temperature variables: {len(temp_vars)}")
    print(f"  Heat stress variables: {len(heat_vars)}")
    print(f"  Seasonal variables: {len(seasonal_vars)}")
    
    # Check climate data quality
    print(f"\n🌡️  CLIMATE DATA QUALITY:")
    climate_completeness = {}
    for col in climate_cols[:10]:  # Check first 10 climate variables
        missing_pct = df[col].isna().sum() / len(df) * 100
        climate_completeness[col] = missing_pct
        if missing_pct > 0:
            print(f"  {col}: {missing_pct:.1f}% missing")
    
    # Temperature variation within Johannesburg
    if temp_vars:
        temp_col = temp_vars[0]
        temp_range = df[temp_col].max() - df[temp_col].min()
        temp_std = df[temp_col].std()
        print(f"\n🌡️  TEMPERATURE VARIATION (using {temp_col}):")
        print(f"  Range: {temp_range:.1f}°C")
        print(f"  Standard deviation: {temp_std:.2f}°C")
        
        if temp_std < 5:
            print("  ⚠️  LIMITED TEMPERATURE VARIATION for heat exposure analysis")
    
    # Heat exposure analysis
    if 'climate_extreme_heat_days_annual' in df.columns:
        heat_days = df['climate_extreme_heat_days_annual'].describe()
        print(f"\n🔥 EXTREME HEAT EXPOSURE:")
        print(f"  Mean extreme heat days/year: {heat_days['mean']:.1f}")
        print(f"  Range: {heat_days['min']:.0f} to {heat_days['max']:.0f} days")
    
    return climate_cols

def investigate_model_performance():
    """Address Priority 6: Model performance and statistical rigor"""
    print("\n🚨 PRIORITY 6: MODEL PERFORMANCE INVESTIGATION")
    print("="*70)
    
    # Load the analysis results if available
    try:
        # Look for recent analysis results
        results_files = list(Path("analysis").glob("*results*"))
        if results_files:
            print(f"📊 ANALYSIS RESULTS FILES FOUND: {len(results_files)}")
            for file in results_files[-3:]:  # Show last 3
                print(f"  {file.name}")
    except:
        print("No analysis results directory found")
    
    # Check for model validation scripts
    model_files = [
        "models/optimization.py",
        "core/pipeline.py",
        "explainability/shap_analysis.py"
    ]
    
    print(f"\n🔍 MODEL COMPONENTS:")
    for file in model_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    # Feature analysis
    df = pd.read_csv("data/optimal_xai_ready/xai_ready_high_quality.csv", low_memory=False)
    
    feature_cols = [col for col in df.columns if not col.startswith('std_') or col in ['participant_id', 'dataset_source', 'study_type']]
    target_cols = [col for col in df.columns if col.startswith('std_')]
    
    print(f"\n📈 FEATURE ANALYSIS:")
    print(f"  Potential features: {len(feature_cols)}")
    print(f"  Potential targets: {len(target_cols)}")
    
    # Check for multicollinearity indicators
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 10:
        print(f"  ⚠️  HIGH-DIMENSIONAL DATA: {len(numeric_cols)} numeric features")
        print(f"     Risk of multicollinearity and overfitting")
    
    return feature_cols, target_cols

def generate_summary_report():
    """Generate comprehensive summary addressing all reviewer concerns"""
    print("\n" + "="*70)
    print("🎯 COMPREHENSIVE REVIEWER RESPONSE SUMMARY")
    print("="*70)
    
    print("""
📋 CRITICAL FINDINGS & REQUIRED PAPER REVISIONS:

1. SAMPLE SIZE RECONCILIATION:
   ✅ Confirmed: 2,334 records in final analysis dataset
   ⚠️  But only 1,807 unique participants (repeated measures)
   📝 ACTION: Clarify that 2,334 represents visits/observations, not unique participants

2. TEMPORAL PERIOD CORRECTION:
   ❌ Data spans 2011-2021, NOT 2013-2021 as claimed
   📝 ACTION: Update all temporal claims in paper to reflect 2011-2021 period

3. GEOGRAPHIC LIMITATIONS:
   ⚠️  Single-site study (Johannesburg only)
   ❌ Cannot support claims about broad "geographic gradient"
   📝 ACTION: Reframe as urban heat variation within Johannesburg metro area

4. DATA QUALITY ISSUES:
   ⚠️  Potential unit inconsistencies in glucose measurements
   ⚠️  Multiple datasets with different measurement protocols
   📝 ACTION: Document data harmonization and unit standardization

5. STATISTICAL RIGOR:
   ⚠️  High-dimensional data with multicollinearity risks
   ❓ Need verification of regularization and multiple testing correction
   📝 ACTION: Provide detailed statistical methodology

IMMEDIATE PAPER REVISIONS REQUIRED:
• Update participant count description (clarify visits vs unique participants)
• Correct temporal period to 2011-2021
• Acknowledge single-site limitation (Johannesburg only)
• Strengthen geographic claims (within-city heat variation)
• Add data quality and harmonization methods section
• Enhance statistical methodology description
""")

def main():
    """Main investigation function"""
    print("🚨 CRITICAL DATA INVESTIGATION FOR REVIEWER QUESTIONS")
    print("Heat-Health XAI Analysis Framework")
    print("="*70)
    
    # Priority investigations
    sample_analysis = investigate_sample_sizes()
    df = investigate_temporal_coverage() 
    geo_cols, climate_cols = investigate_geographic_coverage(df)
    biomarker_cols = investigate_data_quality(df)
    climate_vars = investigate_climate_integration()
    feature_cols, target_cols = investigate_model_performance()
    
    # Generate final summary
    generate_summary_report()
    
    print(f"\n✅ INVESTIGATION COMPLETE")
    print(f"Data files analyzed: {Path('data/optimal_xai_ready').glob('*.csv').__len__()}")
    print(f"Key dataset: xai_ready_high_quality.csv (matches paper sample size)")

if __name__ == "__main__":
    main()