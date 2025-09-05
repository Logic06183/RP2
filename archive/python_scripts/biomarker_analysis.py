#!/usr/bin/env python3
"""
Comprehensive Biomarker Analysis
Detailed examination of health biomarkers with rigorous data quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_health_biomarkers():
    """Comprehensive analysis of health biomarker availability and quality"""
    
    print("Loading health dataset...")
    health_data = pd.read_csv(
        '/home/cparker/incoming/RP2/00_FINAL_DATASETS/HEAT_Johannesburg_FINAL_20250811_163049.csv'
    )
    
    # All glucose-related variables
    glucose_vars = [
        'Glucose (mg/dL)',
        'FASTING GLUCOSE', 
        'Oral glucose tolerance test fasting plasma glucose',
        'Oral glucose tolerance test 1-h plasma glucose',
        'Oral glucose tolerance test 2-h plasma glucose'
    ]
    
    # Comprehensive biomarker list
    key_biomarkers = {
        'Metabolic': {
            'Glucose (mg/dL)': 'Standard glucose',
            'FASTING GLUCOSE': 'Fasting glucose', 
            'Oral glucose tolerance test fasting plasma glucose': 'OGTT fasting glucose',
            'Oral glucose tolerance test 1-h plasma glucose': 'OGTT 1-hour glucose',
            'Oral glucose tolerance test 2-h plasma glucose': 'OGTT 2-hour glucose',
            'FASTING TOTAL CHOLESTEROL': 'Total cholesterol',
            'FASTING HDL': 'HDL cholesterol',
            'FASTING LDL': 'LDL cholesterol', 
            'FASTING TRIGLYCERIDES': 'Triglycerides',
            'FASTING INSULIN': 'Fasting insulin'
        },
        'Cardiovascular': {
            'systolic blood pressure': 'Systolic BP',
            'diastolic blood pressure': 'Diastolic BP',
            'heart rate': 'Heart rate'
        },
        'Hematological': {
            'Hemoglobin (g/dL)': 'Hemoglobin',
            'Hematocrit (%)': 'Hematocrit',
            'White blood cell count (×10³/µL)': 'WBC count',
            'Platelet count (×10³/µL)': 'Platelet count',
            'Lymphocytes (%)': 'Lymphocytes %',
            'Neutrophils (%)': 'Neutrophils %'
        },
        'Renal': {
            'Creatinine (mg/dL)': 'Creatinine',
            'creatinine clearance': 'Creatinine clearance',
            'Blood urea nitrogen (mg/dL)': 'BUN'
        },
        'Hepatic': {
            'ALT (U/L)': 'ALT',
            'AST (U/L)': 'AST',
            'Total bilirubin (mg/dL)': 'Total bilirubin',
            'Alkaline phosphatase (U/L)': 'Alkaline phosphatase'
        },
        'Protein/Nutritional': {
            'Total protein (g/dL)': 'Total protein',
            'Albumin (g/dL)': 'Albumin'
        },
        'Immunological': {
            'CD4 cell count (cells/µL)': 'CD4 count',
            'HIV viral load (copies/mL)': 'HIV viral load'
        }
    }
    
    # Analyze biomarker availability
    biomarker_analysis = {}
    
    for category, markers in key_biomarkers.items():
        category_results = {}
        
        for biomarker, description in markers.items():
            if biomarker in health_data.columns:
                non_null_count = health_data[biomarker].count()
                total_count = len(health_data)
                availability_pct = (non_null_count / total_count) * 100
                
                # Basic statistics for numeric variables
                stats = {}
                if non_null_count > 0:
                    try:
                        numeric_data = pd.to_numeric(health_data[biomarker], errors='coerce')
                        if numeric_data.count() > 0:
                            stats = {
                                'mean': float(numeric_data.mean()),
                                'median': float(numeric_data.median()),
                                'std': float(numeric_data.std()),
                                'min': float(numeric_data.min()),
                                'max': float(numeric_data.max()),
                                'q25': float(numeric_data.quantile(0.25)),
                                'q75': float(numeric_data.quantile(0.75))
                            }
                    except:
                        stats = {'note': 'Non-numeric data'}
                
                category_results[biomarker] = {
                    'description': description,
                    'available_records': int(non_null_count),
                    'total_records': int(total_count),
                    'availability_percentage': round(availability_pct, 2),
                    'statistics': stats
                }
            else:
                category_results[biomarker] = {
                    'description': description,
                    'available_records': 0,
                    'total_records': int(len(health_data)),
                    'availability_percentage': 0.0,
                    'statistics': {'note': 'Column not found'}
                }
        
        biomarker_analysis[category] = category_results
    
    # Study-specific biomarker availability
    study_analysis = {}
    if 'study_source' in health_data.columns:
        for study in health_data['study_source'].unique():
            if pd.isna(study):
                continue
                
            study_data = health_data[health_data['study_source'] == study]
            study_results = {
                'total_records': len(study_data),
                'biomarker_availability': {}
            }
            
            # Check glucose availability by study
            for glucose_var in glucose_vars:
                if glucose_var in health_data.columns:
                    glucose_count = study_data[glucose_var].count()
                    if glucose_count > 0:
                        study_results['biomarker_availability'][glucose_var] = {
                            'count': int(glucose_count),
                            'percentage': round((glucose_count / len(study_data)) * 100, 1)
                        }
            
            # Check other key biomarkers
            key_vars = ['systolic blood pressure', 'diastolic blood pressure', 
                       'FASTING TOTAL CHOLESTEROL', 'CD4 cell count (cells/µL)']
            for var in key_vars:
                if var in health_data.columns:
                    var_count = study_data[var].count()
                    if var_count > 0:
                        study_results['biomarker_availability'][var] = {
                            'count': int(var_count),
                            'percentage': round((var_count / len(study_data)) * 100, 1)
                        }
            
            study_analysis[study] = study_results
    
    # Generate comprehensive report
    report = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'dataset_overview': {
            'total_records': len(health_data),
            'date_range': {
                'start': health_data['primary_date_parsed'].min() if 'primary_date_parsed' in health_data.columns else None,
                'end': health_data['primary_date_parsed'].max() if 'primary_date_parsed' in health_data.columns else None
            },
            'study_count': health_data['study_source'].nunique() if 'study_source' in health_data.columns else None
        },
        'biomarker_analysis_by_category': biomarker_analysis,
        'study_specific_analysis': study_analysis
    }
    
    # Save detailed report
    report_path = Path('/home/cparker/heat_analysis_optimized/biomarker_availability_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate summary
    print("\n=== BIOMARKER AVAILABILITY SUMMARY ===")
    print(f"Total health records: {len(health_data):,}")
    print(f"Total studies: {health_data['study_source'].nunique()}")
    print()
    
    # Glucose availability summary
    print("GLUCOSE VARIABLES:")
    total_glucose_records = 0
    for glucose_var in glucose_vars:
        if glucose_var in health_data.columns:
            count = health_data[glucose_var].count()
            pct = (count / len(health_data)) * 100
            print(f"  - {glucose_var}: {count:,} records ({pct:.1f}%)")
            total_glucose_records += count
    
    print(f"\nTotal glucose measurements: {total_glucose_records:,}")
    print()
    
    # Top studies with glucose data
    print("TOP STUDIES WITH GLUCOSE DATA:")
    glucose_by_study = {}
    for glucose_var in glucose_vars:
        if glucose_var in health_data.columns and health_data[glucose_var].count() > 0:
            study_counts = health_data[health_data[glucose_var].notna()]['study_source'].value_counts()
            for study, count in study_counts.head(5).items():
                if study not in glucose_by_study:
                    glucose_by_study[study] = 0
                glucose_by_study[study] += count
    
    for study, count in sorted(glucose_by_study.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {study}: {count:,} glucose measurements")
    
    print()
    
    # High-availability biomarkers
    print("HIGH-AVAILABILITY BIOMARKERS (>1000 records):")
    high_avail_markers = []
    
    for category, markers in biomarker_analysis.items():
        for marker, info in markers.items():
            if info['available_records'] > 1000:
                high_avail_markers.append((marker, info['available_records'], info['availability_percentage']))
    
    # Sort by availability
    high_avail_markers.sort(key=lambda x: x[1], reverse=True)
    
    for marker, count, pct in high_avail_markers[:10]:
        print(f"  - {marker}: {count:,} records ({pct:.1f}%)")
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    report = analyze_health_biomarkers()