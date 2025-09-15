#!/usr/bin/env python3
"""
Investigate Climate-Biomarker Linkage Issues
============================================
Debug why climate data can't be linked to biomarker records
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.data_loader import DataLoader

def investigate_linkage_issues():
    """Debug the climate-biomarker data linkage problems."""
    print("🔍 INVESTIGATING CLIMATE-BIOMARKER LINKAGE ISSUES")
    print("="*65)
    
    config = Config()
    data_loader = DataLoader(config)
    processed_data = data_loader.process_data()
    
    print(f"📊 Total dataset: {len(processed_data):,} records")
    
    # 1. Check coordinate data availability
    print(f"\\n🌍 COORDINATE DATA ANALYSIS:")
    print("-" * 35)
    
    coord_cols = ['latitude', 'longitude', 'lat', 'lon', 'Latitude', 'Longitude']
    found_coord_cols = []
    
    for col in coord_cols:
        if col in processed_data.columns:
            non_null = processed_data[col].notna().sum()
            print(f"✅ {col}: {non_null:,} records ({non_null/len(processed_data)*100:.1f}%)")
            found_coord_cols.append(col)
            
            # Show sample values
            sample_vals = processed_data[col].dropna().head(10).tolist()
            print(f"   Sample values: {sample_vals}")
        else:
            print(f"❌ {col}: Column not found")
    
    # 2. Check date data availability
    print(f"\\n📅 DATE DATA ANALYSIS:")
    print("-" * 25)
    
    date_cols = ['date', 'Date', 'visit_date', 'enrollment_date', 'collection_date']
    found_date_cols = []
    
    for col in date_cols:
        if col in processed_data.columns:
            non_null = processed_data[col].notna().sum()
            print(f"✅ {col}: {non_null:,} records ({non_null/len(processed_data)*100:.1f}%)")
            found_date_cols.append(col)
            
            # Show sample values and format
            sample_vals = processed_data[col].dropna().head(10).tolist()
            print(f"   Sample values: {sample_vals}")
            print(f"   Data type: {processed_data[col].dtype}")
        else:
            print(f"❌ {col}: Column not found")
    
    # 3. Check biomarker-coordinate overlap
    print(f"\\n🧪 BIOMARKER-COORDINATE OVERLAP:")
    print("-" * 35)
    
    biomarkers = ["systolic blood pressure", "diastolic blood pressure", 
                 "Hemoglobin (g/dL)", "CD4 cell count (cells/µL)", "Creatinine (mg/dL)"]
    
    for biomarker in biomarkers:
        if biomarker in processed_data.columns:
            bio_records = processed_data[processed_data[biomarker].notna()]
            print(f"\\n🔬 {biomarker}: {len(bio_records):,} records")
            
            # Check coordinate availability within biomarker records
            for coord_col in found_coord_cols:
                coord_overlap = bio_records[coord_col].notna().sum()
                overlap_pct = (coord_overlap / len(bio_records)) * 100 if len(bio_records) > 0 else 0
                print(f"   {coord_col}: {coord_overlap:,}/{len(bio_records):,} ({overlap_pct:.1f}%)")
            
            # Check date availability within biomarker records
            for date_col in found_date_cols:
                date_overlap = bio_records[date_col].notna().sum()
                overlap_pct = (date_overlap / len(bio_records)) * 100 if len(bio_records) > 0 else 0
                print(f"   {date_col}: {date_overlap:,}/{len(bio_records):,} ({overlap_pct:.1f}%)")
    
    # 4. Try to find the best coordinate and date columns
    print(f"\\n🎯 RECOMMENDED COLUMNS FOR LINKAGE:")
    print("-" * 40)
    
    if found_coord_cols:
        # Find coordinate columns with best coverage
        coord_coverage = {}
        for col in found_coord_cols:
            coverage = processed_data[col].notna().sum()
            coord_coverage[col] = coverage
        
        best_coord_cols = sorted(coord_coverage.items(), key=lambda x: x[1], reverse=True)
        print(f"🌍 Best coordinate columns:")
        for col, coverage in best_coord_cols[:2]:
            print(f"   ✅ {col}: {coverage:,} records")
    
    if found_date_cols:
        # Find date columns with best coverage
        date_coverage = {}
        for col in found_date_cols:
            coverage = processed_data[col].notna().sum()
            date_coverage[col] = coverage
        
        best_date_cols = sorted(date_coverage.items(), key=lambda x: x[1], reverse=True)
        print(f"📅 Best date columns:")
        for col, coverage in best_date_cols[:2]:
            print(f"   ✅ {col}: {coverage:,} records")
    
    # 5. Check if we can create a linkable subset
    print(f"\\n🔗 POTENTIAL LINKABLE DATASET:")
    print("-" * 32)
    
    if found_coord_cols and found_date_cols:
        # Use best coverage columns
        best_lat = max(found_coord_cols, key=lambda x: processed_data[x].notna().sum()) if found_coord_cols else None
        best_date = max(found_date_cols, key=lambda x: processed_data[x].notna().sum()) if found_date_cols else None
        
        if best_lat and best_date:
            # Find records with both coordinates and dates
            linkable = processed_data[
                processed_data[best_lat].notna() & 
                processed_data[best_date].notna()
            ]
            
            print(f"Records with both {best_lat} and {best_date}: {len(linkable):,}")
            
            # Check biomarker availability in linkable subset
            for biomarker in biomarkers:
                if biomarker in linkable.columns:
                    bio_count = linkable[biomarker].notna().sum()
                    print(f"   {biomarker}: {bio_count:,} records")
            
            # Show sample of linkable data
            if len(linkable) > 0:
                print(f"\\n📋 SAMPLE LINKABLE RECORDS:")
                sample_cols = [best_lat, best_date] + [b for b in biomarkers if b in linkable.columns]
                sample_data = linkable[sample_cols].dropna().head(5)
                print(sample_data.to_string(index=False))
    
    # 6. Look for alternative approaches
    print(f"\\n💡 ALTERNATIVE APPROACHES:")
    print("-" * 28)
    
    # Check if there are ward/municipality columns that could be used for spatial linkage
    spatial_cols = [col for col in processed_data.columns if any(term in col.lower() 
                   for term in ['ward', 'municipality', 'district', 'area', 'region'])]
    
    if spatial_cols:
        print(f"🏙️ Spatial administrative columns available:")
        for col in spatial_cols:
            coverage = processed_data[col].notna().sum()
            unique_vals = processed_data[col].nunique()
            print(f"   ✅ {col}: {coverage:,} records, {unique_vals} unique areas")
    
    # Check for survey year or time period columns
    temporal_cols = [col for col in processed_data.columns if any(term in col.lower() 
                    for term in ['year', 'survey', 'wave', 'period', 'time'])]
    
    if temporal_cols:
        print(f"📅 Temporal period columns available:")
        for col in temporal_cols:
            coverage = processed_data[col].notna().sum()
            unique_vals = processed_data[col].nunique()
            sample_vals = processed_data[col].dropna().unique()[:5]
            print(f"   ✅ {col}: {coverage:,} records, {unique_vals} periods")
            print(f"      Sample periods: {sample_vals.tolist()}")

if __name__ == "__main__":
    investigate_linkage_issues()