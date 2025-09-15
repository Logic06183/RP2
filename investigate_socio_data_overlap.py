#!/usr/bin/env python3
"""
Investigate Socioeconomic Data Overlap Issue
===========================================
Debug why GCRO socioeconomic variables aren't appearing with biomarkers.
"""

import pandas as pd
import numpy as np

def investigate_overlap():
    """Investigate the overlap between socioeconomic and biomarker data."""
    print("🔍 INVESTIGATING SOCIOECONOMIC-BIOMARKER DATA OVERLAP")
    print("="*65)
    
    # Load data
    data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/ENHANCED_CLIMATE_BIOMARKER_DATASET.csv"
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} total records")
    except:
        print("❌ Could not load data")
        return
    
    # Check socioeconomic variables
    socio_vars = ['employment', 'education', 'income', 'municipality', 'ward']
    biomarker_vars = ['systolic blood pressure', 'diastolic blood pressure', 
                      'CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)', 'Creatinine (mg/dL)']
    
    print(f"\n📊 OVERALL DATA COVERAGE:")
    print("-" * 30)
    
    for var in socio_vars:
        if var in df.columns:
            count = df[var].notna().sum()
            pct = count / len(df) * 100
            print(f"{var}: {count:,} ({pct:.1f}%)")
    
    for var in biomarker_vars:
        if var in df.columns:
            count = df[var].notna().sum()
            pct = count / len(df) * 100
            print(f"{var}: {count:,} ({pct:.1f}%)")
    
    # Critical test: Check overlap
    print(f"\n🔍 OVERLAP ANALYSIS:")
    print("-" * 25)
    
    # Records with socioeconomic data
    has_socio = df[socio_vars].notna().any(axis=1)
    socio_records = df[has_socio]
    print(f"Records with ANY socioeconomic data: {len(socio_records):,}")
    
    # Records with biomarker data
    has_bio = df[biomarker_vars].notna().any(axis=1)
    bio_records = df[has_bio]
    print(f"Records with ANY biomarker data: {len(bio_records):,}")
    
    # The critical overlap
    has_both = has_socio & has_bio
    overlap_records = df[has_both]
    print(f"Records with BOTH socioeconomic AND biomarker data: {len(overlap_records):,}")
    
    if len(overlap_records) == 0:
        print("\n🚨 PROBLEM IDENTIFIED: NO OVERLAP!")
        print("The GCRO socioeconomic data and biomarker data are in separate records!")
        
        # Investigate why
        print(f"\n🔍 INVESTIGATING WHY NO OVERLAP:")
        print("-" * 35)
        
        # Check data sources
        print("Socioeconomic data sources:")
        socio_sample = socio_records.head(5)
        for col in ['latitude', 'longitude', 'survey_wave', 'survey_year']:
            if col in socio_sample.columns:
                print(f"  {col}: {socio_sample[col].tolist()}")
        
        print("\nBiomarker data sources:")
        bio_sample = bio_records.head(5)
        for col in ['latitude', 'longitude', 'survey_wave', 'survey_year']:
            if col in bio_sample.columns:
                print(f"  {col}: {bio_sample[col].tolist()}")
                
        # The solution
        print(f"\n💡 SOLUTION:")
        print("Need to create synthetic overlap by matching geographic/temporal patterns")
        print("or use broader inclusion criteria that doesn't require exact overlap")
        
    else:
        print(f"\n✅ Found {len(overlap_records):,} records with both types of data")
        
        # Analyze the overlap
        print(f"\nOverlap analysis by biomarker:")
        for bio_var in biomarker_vars:
            if bio_var in df.columns:
                overlap_with_bio = overlap_records[bio_var].notna().sum()
                print(f"  {bio_var}: {overlap_with_bio} records")
        
        print(f"\nOverlap analysis by socioeconomic variable:")
        for socio_var in socio_vars:
            if socio_var in df.columns:
                overlap_with_socio = overlap_records[socio_var].notna().sum()
                print(f"  {socio_var}: {overlap_with_socio} records")

if __name__ == "__main__":
    investigate_overlap()