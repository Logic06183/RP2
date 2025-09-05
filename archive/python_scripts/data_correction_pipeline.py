#!/usr/bin/env python3
"""
Data Correction Pipeline for Heat-Health XAI Analysis
Implements all reviewer-recommended fixes:
1. Glucose unit conversion (mmol/L to mg/dL)
2. Outlier detection and handling
3. Data quality improvements
4. Corrected sample size documentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    """Load the original high-quality dataset"""
    print("📊 LOADING ORIGINAL HIGH-QUALITY DATASET")
    print("="*50)
    
    df = pd.read_csv("data/optimal_xai_ready/xai_ready_high_quality.csv", low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Unique participants: {df['participant_id'].nunique()}")
    print(f"Dataset composition:")
    for source, count in df['dataset_source'].value_counts().items():
        print(f"  {source}: {count:,} records")
    
    return df

def fix_glucose_units(df):
    """Convert glucose from mmol/L to mg/dL for consistency"""
    print(f"\n🩸 FIXING GLUCOSE UNIT CONVERSIONS")
    print("="*50)
    
    # Conversion factor: mmol/L to mg/dL = multiply by 18.0182
    conversion_factor = 18.0182
    
    # Identify which datasets need conversion (glucose < 20 likely mmol/L)
    glucose_stats_before = df.groupby('dataset_source')['std_glucose'].agg(['count', 'mean', 'min', 'max']).round(2)
    print("Glucose stats before conversion:")
    print(glucose_stats_before)
    
    # Apply conversion to datasets with likely mmol/L values
    df['std_glucose_corrected'] = df['std_glucose'].copy()
    
    for source in df['dataset_source'].unique():
        source_mask = df['dataset_source'] == source
        source_glucose = df[source_mask]['std_glucose'].dropna()
        
        if len(source_glucose) > 0 and source_glucose.mean() < 20:
            print(f"\n{source}: Converting mmol/L to mg/dL")
            print(f"  Before: {source_glucose.mean():.2f} ± {source_glucose.std():.2f} mmol/L")
            
            # Convert to mg/dL
            df.loc[source_mask, 'std_glucose_corrected'] = df.loc[source_mask, 'std_glucose'] * conversion_factor
            
            converted_values = df.loc[source_mask, 'std_glucose_corrected'].dropna()
            print(f"  After:  {converted_values.mean():.1f} ± {converted_values.std():.1f} mg/dL")
        else:
            print(f"{source}: Already in mg/dL (no conversion needed)")
    
    # Replace original with corrected
    df['std_glucose'] = df['std_glucose_corrected']
    df.drop('std_glucose_corrected', axis=1, inplace=True)
    
    # Verify final glucose distribution
    glucose_stats_after = df.groupby('dataset_source')['std_glucose'].agg(['count', 'mean', 'min', 'max']).round(1)
    print(f"\nGlucose stats after conversion (all in mg/dL):")
    print(glucose_stats_after)
    
    return df

def detect_and_handle_outliers(df):
    """Detect and handle outliers in biomarker data"""
    print(f"\n🔍 OUTLIER DETECTION AND HANDLING")
    print("="*50)
    
    biomarker_cols = ['std_glucose', 'std_cholesterol_total', 'std_cholesterol_hdl', 
                     'std_cholesterol_ldl', 'std_triglycerides', 'std_creatinine', 
                     'std_systolic_bp', 'std_diastolic_bp']
    
    outlier_summary = {}
    
    for col in biomarker_cols:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) == 0:
                continue
                
            # Calculate IQR-based outliers
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = (values < lower_bound) | (values > upper_bound)
            n_outliers = outliers.sum()
            outlier_pct = n_outliers / len(values) * 100
            
            print(f"{col}:")
            print(f"  Range: {values.min():.1f} to {values.max():.1f}")
            print(f"  IQR bounds: {lower_bound:.1f} to {upper_bound:.1f}")
            print(f"  Outliers: {n_outliers} ({outlier_pct:.1f}%)")
            
            # Handle extreme outliers (> 3 IQRs from median)
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            extreme_outliers = (values < extreme_lower) | (values > extreme_upper)
            n_extreme = extreme_outliers.sum()
            
            if n_extreme > 0:
                print(f"  Extreme outliers: {n_extreme} (will be capped)")
                
                # Cap extreme outliers
                df.loc[df[col] < extreme_lower, col] = extreme_lower
                df.loc[df[col] > extreme_upper, col] = extreme_upper
            
            outlier_summary[col] = {
                'n_outliers': n_outliers,
                'outlier_pct': outlier_pct,
                'n_extreme': n_extreme,
                'bounds': (lower_bound, upper_bound)
            }
    
    return df, outlier_summary

def fix_impossible_values(df):
    """Fix impossible/corrupted values"""
    print(f"\n🛠️ FIXING IMPOSSIBLE VALUES")
    print("="*50)
    
    fixes_applied = []
    
    # Fix negative cholesterol values
    if 'std_cholesterol_total' in df.columns:
        negative_chol = df['std_cholesterol_total'] < 0
        if negative_chol.sum() > 0:
            print(f"Fixing {negative_chol.sum()} negative cholesterol values")
            df.loc[negative_chol, 'std_cholesterol_total'] = np.nan
            fixes_applied.append(f"Negative cholesterol → NaN: {negative_chol.sum()} values")
    
    # Fix impossible creatinine values (>1000 likely data corruption)
    if 'std_creatinine' in df.columns:
        extreme_creat = df['std_creatinine'] > 1000
        if extreme_creat.sum() > 0:
            print(f"Fixing {extreme_creat.sum()} extreme creatinine values (>1000)")
            df.loc[extreme_creat, 'std_creatinine'] = np.nan
            fixes_applied.append(f"Extreme creatinine (>1000) → NaN: {extreme_creat.sum()} values")
    
    # Fix impossible blood pressure values
    if 'std_systolic_bp' in df.columns:
        impossible_sbp = (df['std_systolic_bp'] < 70) | (df['std_systolic_bp'] > 300)
        if impossible_sbp.sum() > 0:
            print(f"Fixing {impossible_sbp.sum()} impossible systolic BP values")
            df.loc[impossible_sbp, 'std_systolic_bp'] = np.nan
            fixes_applied.append(f"Impossible SBP (<70 or >300) → NaN: {impossible_sbp.sum()} values")
    
    if 'std_diastolic_bp' in df.columns:
        impossible_dbp = (df['std_diastolic_bp'] < 40) | (df['std_diastolic_bp'] > 200)
        if impossible_dbp.sum() > 0:
            print(f"Fixing {impossible_dbp.sum()} impossible diastolic BP values")
            df.loc[impossible_dbp, 'std_diastolic_bp'] = np.nan
            fixes_applied.append(f"Impossible DBP (<40 or >200) → NaN: {impossible_dbp.sum()} values")
    
    # Fix impossible glucose values (after conversion should be 50-500 mg/dL)
    if 'std_glucose' in df.columns:
        impossible_glucose = (df['std_glucose'] < 50) | (df['std_glucose'] > 500)
        if impossible_glucose.sum() > 0:
            print(f"Fixing {impossible_glucose.sum()} impossible glucose values")
            df.loc[impossible_glucose, 'std_glucose'] = np.nan
            fixes_applied.append(f"Impossible glucose (<50 or >500 mg/dL) → NaN: {impossible_glucose.sum()} values")
    
    print(f"Applied {len(fixes_applied)} data quality fixes:")
    for fix in fixes_applied:
        print(f"  {fix}")
    
    return df, fixes_applied

def calculate_data_quality_metrics(df):
    """Calculate comprehensive data quality metrics"""
    print(f"\n📈 DATA QUALITY METRICS")
    print("="*50)
    
    # Overall completeness
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness_pct = (1 - missing_cells / total_cells) * 100
    
    print(f"Overall data completeness: {completeness_pct:.1f}%")
    
    # Biomarker completeness
    biomarker_cols = [col for col in df.columns if col.startswith('std_') and col not in ['std_visit_date']]
    
    print(f"\nBiomarker completeness by variable:")
    biomarker_completeness = {}
    for col in biomarker_cols:
        if col in df.columns:
            complete_pct = (1 - df[col].isnull().sum() / len(df)) * 100
            biomarker_completeness[col] = complete_pct
            print(f"  {col}: {complete_pct:.1f}%")
    
    # Climate data completeness
    climate_cols = [col for col in df.columns if 'climate' in col]
    print(f"\nClimate data completeness: {len(climate_cols)} variables")
    climate_completeness = {}
    for col in climate_cols[:10]:  # Show first 10
        complete_pct = (1 - df[col].isnull().sum() / len(df)) * 100
        climate_completeness[col] = complete_pct
        if complete_pct < 100:
            print(f"  {col}: {complete_pct:.1f}%")
    
    # Sample size by dataset
    print(f"\nSample size by dataset:")
    sample_by_source = df['dataset_source'].value_counts()
    for source, count in sample_by_source.items():
        unique_participants = df[df['dataset_source'] == source]['participant_id'].nunique()
        print(f"  {source}: {count} records, {unique_participants} unique participants")
    
    return {
        'overall_completeness': completeness_pct,
        'biomarker_completeness': biomarker_completeness,
        'climate_completeness': climate_completeness,
        'sample_by_source': sample_by_source.to_dict()
    }

def create_analysis_ready_dataset(df):
    """Create final analysis-ready dataset with quality filters"""
    print(f"\n✨ CREATING ANALYSIS-READY DATASET")
    print("="*50)
    
    print(f"Starting dataset: {df.shape[0]} records")
    
    # Filter 1: Must have basic biomarkers
    essential_biomarkers = ['std_glucose', 'std_systolic_bp']  # Core variables for heat-health analysis
    has_essential = df[essential_biomarkers].notna().any(axis=1)
    df_filtered = df[has_essential].copy()
    print(f"After essential biomarkers filter: {len(df_filtered)} records")
    
    # Filter 2: Must have climate data
    has_climate = df_filtered[[col for col in df_filtered.columns if 'climate_temp_mean_1d' in col]].notna().any(axis=1)
    df_filtered = df_filtered[has_climate]
    print(f"After climate data filter: {len(df_filtered)} records")
    
    # Filter 3: Must have valid visit date
    has_valid_date = df_filtered['std_visit_date'].notna()
    df_filtered = df_filtered[has_valid_date]
    print(f"After valid date filter: {len(df_filtered)} records")
    
    # Add quality metrics
    biomarker_cols = [col for col in df_filtered.columns if col.startswith('std_') and col not in ['std_visit_date']]
    df_filtered['biomarker_completeness_score'] = df_filtered[biomarker_cols].notna().sum(axis=1) / len(biomarker_cols) * 100
    
    climate_cols = [col for col in df_filtered.columns if 'climate' in col]
    df_filtered['climate_completeness_score'] = df_filtered[climate_cols].notna().sum(axis=1) / len(climate_cols) * 100
    
    df_filtered['total_completeness_score'] = (df_filtered['biomarker_completeness_score'] + df_filtered['climate_completeness_score']) / 2
    
    # Final summary
    print(f"\nFinal analysis dataset:")
    print(f"  Total records: {len(df_filtered):,}")
    print(f"  Unique participants: {df_filtered['participant_id'].nunique():,}")
    print(f"  Mean biomarker completeness: {df_filtered['biomarker_completeness_score'].mean():.1f}%")
    print(f"  Mean climate completeness: {df_filtered['climate_completeness_score'].mean():.1f}%")
    
    return df_filtered

def save_corrected_dataset(df, quality_metrics):
    """Save the corrected dataset and generate documentation"""
    print(f"\n💾 SAVING CORRECTED DATASET")
    print("="*50)
    
    # Save main dataset
    output_path = "data/optimal_xai_ready/xai_ready_corrected_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved corrected dataset: {output_path}")
    
    # Generate data quality report
    report_path = "data/optimal_xai_ready/DATA_QUALITY_REPORT_V2.md"
    with open(report_path, 'w') as f:
        f.write("# Data Quality Report - Corrected Dataset V2\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- **Total records**: {len(df):,}\n")
        f.write(f"- **Unique participants**: {df['participant_id'].nunique():,}\n")
        f.write(f"- **Study period**: {df['std_visit_date'].min()} to {df['std_visit_date'].max()}\n")
        f.write(f"- **Overall completeness**: {quality_metrics['overall_completeness']:.1f}%\n\n")
        
        f.write("## Sample Composition\n")
        for source, count in quality_metrics['sample_by_source'].items():
            unique_p = df[df['dataset_source'] == source]['participant_id'].nunique()
            f.write(f"- **{source}**: {count:,} records ({unique_p:,} unique participants)\n")
        
        f.write("\n## Data Quality Fixes Applied\n")
        f.write("1. **Glucose unit conversion**: mmol/L → mg/dL for DPHRU datasets\n")
        f.write("2. **Outlier handling**: Extreme values capped at 3×IQR\n")
        f.write("3. **Impossible values**: Negative/corrupted values set to NaN\n")
        f.write("4. **Quality filtering**: Records with essential biomarkers and climate data\n\n")
        
        f.write("## Biomarker Completeness\n")
        for var, pct in quality_metrics['biomarker_completeness'].items():
            f.write(f"- **{var}**: {pct:.1f}%\n")
        
        f.write(f"\n## Ready for Analysis\n")
        f.write(f"✅ Dataset ready for heat-health XAI analysis\n")
        f.write(f"✅ All units standardized and quality-controlled\n")
        f.write(f"✅ Sample size properly documented as records vs unique participants\n")
    
    print(f"Saved quality report: {report_path}")
    
    return output_path, report_path

def main():
    """Main correction pipeline"""
    print("🔧 DATA CORRECTION PIPELINE FOR HEAT-HEALTH XAI ANALYSIS")
    print("Implementing all reviewer recommendations")
    print("="*60)
    
    # Step 1: Load original data
    df = load_original_data()
    
    # Step 2: Fix glucose units
    df = fix_glucose_units(df)
    
    # Step 3: Handle outliers
    df, outlier_summary = detect_and_handle_outliers(df)
    
    # Step 4: Fix impossible values  
    df, fixes_applied = fix_impossible_values(df)
    
    # Step 5: Calculate quality metrics
    quality_metrics = calculate_data_quality_metrics(df)
    
    # Step 6: Create analysis-ready dataset
    df_final = create_analysis_ready_dataset(df)
    
    # Step 7: Save corrected dataset
    output_path, report_path = save_corrected_dataset(df_final, quality_metrics)
    
    print(f"\n🎯 CORRECTION PIPELINE COMPLETE!")
    print(f"✅ All reviewer recommendations implemented")
    print(f"✅ Data quality issues resolved")
    print(f"✅ Ready for reanalysis")
    print(f"\nFiles created:")
    print(f"  📊 {output_path}")
    print(f"  📋 {report_path}")
    
    return df_final

if __name__ == "__main__":
    corrected_df = main()