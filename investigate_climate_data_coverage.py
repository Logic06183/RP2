#!/usr/bin/env python3
"""
Climate Data Coverage Investigation
==================================
Deep dive into why climate variables show no SHAP importance
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.data_loader import DataLoader

def investigate_climate_coverage():
    """Investigate climate data coverage and quality issues."""
    print("🔍 CLIMATE DATA COVERAGE INVESTIGATION")
    print("="*70)
    
    config = Config()
    data_loader = DataLoader(config)
    
    # Load raw data without processing
    print("📊 Loading master dataset...")
    processed_data = data_loader.process_data()
    
    print(f"Total dataset: {len(processed_data):,} records")
    
    # Find all climate-related columns
    climate_patterns = ['temp', 'heat', 'cool', 'era5', 'climate', 'weather']
    climate_cols = []
    
    for col in processed_data.columns:
        if any(pattern in col.lower() for pattern in climate_patterns):
            climate_cols.append(col)
    
    print(f"\n🌡️ FOUND {len(climate_cols)} CLIMATE-RELATED COLUMNS:")
    print("-" * 60)
    
    # Detailed analysis of each climate variable
    climate_analysis = []
    
    for col in climate_cols:
        non_null = processed_data[col].notna().sum()
        coverage_pct = (non_null / len(processed_data)) * 100
        
        if non_null > 0:
            unique_vals = processed_data[col].nunique()
            dtype = processed_data[col].dtype
            
            # Sample non-null values
            sample_vals = processed_data[col].dropna().head(10).tolist()
            
            # Basic statistics for numeric columns
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                mean_val = processed_data[col].mean()
                std_val = processed_data[col].std()
                min_val = processed_data[col].min()
                max_val = processed_data[col].max()
                stats = f"μ={mean_val:.2f}, σ={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]"
            else:
                stats = f"Most common: {processed_data[col].value_counts().head(3).to_dict()}"
            
            climate_analysis.append({
                'column': col,
                'coverage': non_null,
                'coverage_pct': coverage_pct,
                'unique_vals': unique_vals,
                'dtype': dtype,
                'sample_vals': sample_vals,
                'stats': stats
            })
    
    # Sort by coverage
    climate_analysis.sort(key=lambda x: x['coverage'], reverse=True)
    
    for i, info in enumerate(climate_analysis, 1):
        print(f"\n{i:2d}. {info['column']}")
        print(f"    📊 Coverage: {info['coverage']:,} records ({info['coverage_pct']:.1f}%)")
        print(f"    🔢 Data Type: {info['dtype']}")
        print(f"    🎯 Unique Values: {info['unique_vals']:,}")
        print(f"    📈 Statistics: {info['stats']}")
        print(f"    🔍 Sample Values: {info['sample_vals']}")
        
        # Special analysis for high-coverage columns
        if info['coverage_pct'] > 80:
            print(f"    ⭐ HIGH COVERAGE - Good for modeling!")
        elif info['coverage_pct'] > 10:
            print(f"    🟡 MODERATE COVERAGE - Limited utility")
        else:
            print(f"    🔴 LOW COVERAGE - Poor for modeling")
    
    # Look for biomarker records with climate data
    print(f"\n🧪 BIOMARKER-CLIMATE DATA OVERLAP:")
    print("-" * 40)
    
    biomarkers = ["systolic blood pressure", "diastolic blood pressure", 
                 "Hemoglobin (g/dL)", "CD4 cell count (cells/µL)", "Creatinine (mg/dL)"]
    
    for biomarker in biomarkers:
        if biomarker in processed_data.columns:
            bio_records = processed_data[processed_data[biomarker].notna()]
            print(f"\n🔬 {biomarker}: {len(bio_records):,} records")
            
            # Check climate coverage within biomarker records
            for info in climate_analysis[:5]:  # Top 5 climate variables
                col = info['column']
                climate_in_bio = bio_records[col].notna().sum()
                overlap_pct = (climate_in_bio / len(bio_records)) * 100 if len(bio_records) > 0 else 0
                
                print(f"   • {col}: {climate_in_bio:,}/{len(bio_records):,} ({overlap_pct:.1f}%)")
                
                # If good overlap, show some actual values
                if overlap_pct > 5:
                    sample_climate_bio = bio_records[col].dropna().head(5)
                    print(f"     Sample values: {sample_climate_bio.tolist()}")
    
    # Investigate the q1_19_4_heat variable specifically
    print(f"\n🔥 INVESTIGATING q1_19_4_heat (100% coverage):")
    print("-" * 50)
    
    if 'q1_19_4_heat' in processed_data.columns:
        heat_var = processed_data['q1_19_4_heat']
        
        print(f"Data type: {heat_var.dtype}")
        print(f"Unique values: {heat_var.nunique()}")
        print(f"Value counts:")
        print(heat_var.value_counts().head(10))
        
        # Check if this is actually useful for modeling
        if heat_var.nunique() <= 10:
            print("⚠️ This appears to be a categorical survey response, not climate data!")
            print("   Likely a questionnaire item about heat perception/experience")
        else:
            print("✅ This might be useful continuous climate data")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    high_coverage_climate = [info for info in climate_analysis if info['coverage_pct'] > 80]
    
    if high_coverage_climate:
        print("✅ Use these high-coverage climate variables:")
        for info in high_coverage_climate:
            print(f"   • {info['column']} ({info['coverage_pct']:.1f}% coverage)")
    else:
        print("❌ No high-coverage climate variables found!")
    
    print(f"\n🚨 LIKELY ISSUES:")
    print("1. ERA5 climate data appears to be a small sample (500 records)")
    print("2. Most climate data doesn't overlap with biomarker records") 
    print("3. q1_19_4_heat is likely a survey question, not actual climate data")
    print("4. Need proper climate-biomarker data integration")

if __name__ == "__main__":
    investigate_climate_coverage()