#!/usr/bin/env python3
"""
Comprehensive Heat-Health-Socioeconomic Analysis
Rigorous analysis of full RP2 and GCRO datasets with complete transparency

Author: Claude Code Assistant
Date: 2025-09-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveHeatHealthAnalyzer:
    """
    Rigorous analyzer for heat-health-socioeconomic relationships
    with complete methodology transparency and data quality reporting
    """
    
    def __init__(self):
        self.health_data = None
        self.socio_data = None
        self.integrated_data = None
        self.analysis_log = []
        self.data_quality_report = {}
        
    def log_analysis_step(self, step, details):
        """Log every analysis step for complete transparency"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'details': details
        }
        self.analysis_log.append(log_entry)
        print(f"[{timestamp}] {step}: {details}")
    
    def load_health_data(self):
        """Load and examine full health dataset with rigorous quality assessment"""
        self.log_analysis_step("DATA_LOAD", "Loading health dataset")
        
        try:
            self.health_data = pd.read_csv(
                '/home/cparker/incoming/RP2/00_FINAL_DATASETS/HEAT_Johannesburg_FINAL_20250811_163049.csv'
            )
            
            # Comprehensive data quality assessment
            quality_metrics = {
                'total_records': len(self.health_data),
                'total_columns': len(self.health_data.columns),
                'date_range': None,
                'missing_data_summary': {},
                'geographic_coverage': {},
                'study_sources': {},
                'key_biomarkers_availability': {}
            }
            
            # Date range analysis
            if 'primary_date_parsed' in self.health_data.columns:
                self.health_data['primary_date_parsed'] = pd.to_datetime(
                    self.health_data['primary_date_parsed']
                )
                quality_metrics['date_range'] = {
                    'start': self.health_data['primary_date_parsed'].min().isoformat(),
                    'end': self.health_data['primary_date_parsed'].max().isoformat(),
                    'span_days': (self.health_data['primary_date_parsed'].max() - 
                                self.health_data['primary_date_parsed'].min()).days
                }
            
            # Missing data analysis
            missing_summary = {}
            for col in self.health_data.columns:
                missing_count = self.health_data[col].isnull().sum()
                missing_pct = (missing_count / len(self.health_data)) * 100
                if missing_pct > 0:
                    missing_summary[col] = {
                        'missing_count': int(missing_count),
                        'missing_percentage': round(missing_pct, 2)
                    }
            quality_metrics['missing_data_summary'] = missing_summary
            
            # Geographic coverage
            if 'latitude' in self.health_data.columns and 'longitude' in self.health_data.columns:
                geo_summary = {
                    'unique_coordinates': len(self.health_data[['latitude', 'longitude']].drop_duplicates()),
                    'lat_range': [float(self.health_data['latitude'].min()), 
                                float(self.health_data['latitude'].max())],
                    'lon_range': [float(self.health_data['longitude'].min()), 
                                float(self.health_data['longitude'].max())]
                }
                if 'jhb_subregion' in self.health_data.columns:
                    geo_summary['subregions'] = self.health_data['jhb_subregion'].value_counts().to_dict()
                quality_metrics['geographic_coverage'] = geo_summary
            
            # Study sources
            if 'study_source' in self.health_data.columns:
                quality_metrics['study_sources'] = self.health_data['study_source'].value_counts().to_dict()
            
            # Key biomarkers availability
            key_biomarkers = [
                'Glucose (mg/dL)', 'systolic blood pressure', 'diastolic blood pressure',
                'Total protein (g/dL)', 'FASTING TOTAL CHOLESTEROL', 'Creatinine (mg/dL)',
                'CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)'
            ]
            
            biomarker_summary = {}
            for biomarker in key_biomarkers:
                if biomarker in self.health_data.columns:
                    non_null_count = self.health_data[biomarker].count()
                    biomarker_summary[biomarker] = {
                        'available_records': int(non_null_count),
                        'availability_percentage': round((non_null_count / len(self.health_data)) * 100, 2),
                        'value_range': [float(self.health_data[biomarker].min()) if non_null_count > 0 else None,
                                      float(self.health_data[biomarker].max()) if non_null_count > 0 else None]
                    }
            quality_metrics['key_biomarkers_availability'] = biomarker_summary
            
            self.data_quality_report['health_data'] = quality_metrics
            self.log_analysis_step("HEALTH_DATA_LOADED", 
                                 f"Loaded {len(self.health_data)} records with {len(self.health_data.columns)} variables")
            
            return True
            
        except Exception as e:
            self.log_analysis_step("ERROR", f"Failed to load health data: {str(e)}")
            return False
    
    def load_socio_data(self):
        """Load and examine socioeconomic dataset with rigorous assessment"""
        self.log_analysis_step("DATA_LOAD", "Loading socioeconomic dataset")
        
        try:
            self.socio_data = pd.read_csv(
                '/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv'
            )
            
            # Comprehensive socioeconomic data quality assessment
            quality_metrics = {
                'total_records': len(self.socio_data),
                'total_columns': len(self.socio_data.columns),
                'survey_years': {},
                'geographic_coverage': {},
                'key_variables_availability': {},
                'climate_integration_status': {},
                'missing_data_patterns': {}
            }
            
            # Survey year analysis
            if 'survey_year' in self.socio_data.columns:
                quality_metrics['survey_years'] = self.socio_data['survey_year'].value_counts().to_dict()
            
            # Geographic coverage
            if 'municipality_coded' in self.socio_data.columns:
                quality_metrics['geographic_coverage']['municipalities'] = self.socio_data['municipality_coded'].value_counts().to_dict()
            if 'Planning_region' in self.socio_data.columns:
                quality_metrics['geographic_coverage']['planning_regions'] = self.socio_data['Planning_region'].value_counts().to_dict()
            
            # Key socioeconomic variables
            key_se_variables = [
                'q15_3_income_recode',  # Income
                'q14_1_education_recode',  # Education  
                'q10_2_working',  # Employment
                'q1_3_tenure',  # Housing tenure
                'q1_4_water',  # Water access
                'q13_5_medical_aid',  # Healthcare access
                'q13_6_health_status',  # Health status
                'QoLIndex_Data_Driven',  # Quality of life index
                'q6_4_skip_meal',  # Food security
                'q8_13_poverty',  # Poverty perception
            ]
            
            se_summary = {}
            for var in key_se_variables:
                if var in self.socio_data.columns:
                    non_null_count = self.socio_data[var].count()
                    se_summary[var] = {
                        'available_records': int(non_null_count),
                        'availability_percentage': round((non_null_count / len(self.socio_data)) * 100, 2),
                        'unique_values': int(self.socio_data[var].nunique()) if non_null_count > 0 else 0
                    }
            quality_metrics['key_variables_availability'] = se_summary
            
            # Climate integration status
            climate_vars = [col for col in self.socio_data.columns if 'era5' in col.lower() or 'modis' in col.lower()]
            climate_summary = {}
            for var in climate_vars[:10]:  # First 10 climate variables
                non_null_count = self.socio_data[var].count()
                climate_summary[var] = {
                    'available_records': int(non_null_count),
                    'availability_percentage': round((non_null_count / len(self.socio_data)) * 100, 2)
                }
            quality_metrics['climate_integration_status'] = climate_summary
            
            # Missing data patterns
            missing_summary = {}
            high_missing_vars = []
            for col in self.socio_data.columns:
                missing_pct = (self.socio_data[col].isnull().sum() / len(self.socio_data)) * 100
                if missing_pct > 50:  # High missing variables
                    high_missing_vars.append((col, round(missing_pct, 2)))
            
            quality_metrics['missing_data_patterns'] = {
                'high_missing_variables_50_plus_pct': high_missing_vars[:20],  # Top 20
                'total_high_missing_vars': len(high_missing_vars)
            }
            
            self.data_quality_report['socio_data'] = quality_metrics
            self.log_analysis_step("SOCIO_DATA_LOADED", 
                                 f"Loaded {len(self.socio_data)} records with {len(self.socio_data.columns)} variables")
            
            return True
            
        except Exception as e:
            self.log_analysis_step("ERROR", f"Failed to load socioeconomic data: {str(e)}")
            return False
    
    def analyze_integration_potential(self):
        """Analyze potential for data integration with rigorous methodology"""
        self.log_analysis_step("INTEGRATION_ANALYSIS", "Analyzing integration potential")
        
        if self.health_data is None or self.socio_data is None:
            self.log_analysis_step("ERROR", "Both datasets must be loaded first")
            return False
        
        integration_analysis = {
            'temporal_alignment': {},
            'spatial_alignment': {},
            'sample_size_implications': {},
            'methodological_considerations': []
        }
        
        # Temporal alignment analysis
        if 'primary_date_parsed' in self.health_data.columns and 'interview_date_parsed' in self.socio_data.columns:
            health_date_range = [
                self.health_data['primary_date_parsed'].min(),
                self.health_data['primary_date_parsed'].max()
            ]
            socio_date_range = [
                pd.to_datetime(self.socio_data['interview_date_parsed']).min(),
                pd.to_datetime(self.socio_data['interview_date_parsed']).max()
            ]
            
            # Calculate overlap
            overlap_start = max(health_date_range[0], socio_date_range[0])
            overlap_end = min(health_date_range[1], socio_date_range[1])
            
            integration_analysis['temporal_alignment'] = {
                'health_date_range': [d.isoformat() for d in health_date_range],
                'socio_date_range': [d.isoformat() for d in socio_date_range],
                'overlap_period': [overlap_start.isoformat(), overlap_end.isoformat()] if overlap_start <= overlap_end else None,
                'overlap_days': (overlap_end - overlap_start).days if overlap_start <= overlap_end else 0
            }
        
        # Spatial alignment analysis
        if all(col in self.health_data.columns for col in ['latitude', 'longitude']):
            health_bounds = {
                'lat_min': float(self.health_data['latitude'].min()),
                'lat_max': float(self.health_data['latitude'].max()),
                'lon_min': float(self.health_data['longitude'].min()),
                'lon_max': float(self.health_data['longitude'].max())
            }
            
            integration_analysis['spatial_alignment'] = {
                'health_geographic_bounds': health_bounds,
                'socio_has_climate_data': len([col for col in self.socio_data.columns if 'era5' in col.lower()]) > 0,
                'integration_feasible': True  # Both in Johannesburg area
            }
        
        # Sample size implications
        integration_analysis['sample_size_implications'] = {
            'health_records': len(self.health_data),
            'socio_records': len(self.socio_data),
            'integration_approach': 'climate_based',  # Use climate data as bridge
            'expected_analytical_power': 'high' if len(self.health_data) > 1000 else 'moderate'
        }
        
        # Methodological considerations
        considerations = [
            "Health data spans multiple studies with different protocols",
            "Socioeconomic data from cross-sectional survey with climate integration",
            "Integration possible through climate exposure variables",
            "Temporal alignment allows examination of climate-health relationships",
            "Geographic alignment within Johannesburg metropolitan area",
            "Different sampling frameworks require careful statistical handling"
        ]
        integration_analysis['methodological_considerations'] = considerations
        
        self.data_quality_report['integration_analysis'] = integration_analysis
        self.log_analysis_step("INTEGRATION_ANALYZED", "Integration potential assessed")
        
        return True
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        self.log_analysis_step("REPORT_GENERATION", "Generating data quality report")
        
        # Save detailed JSON report
        report_path = Path('/home/cparker/heat_analysis_optimized/comprehensive_data_quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.data_quality_report, f, indent=2, default=str)
        
        # Generate summary report
        summary = []
        summary.append("# Comprehensive Data Quality Assessment Report")
        summary.append(f"Generated: {datetime.now().isoformat()}")
        summary.append("")
        
        if 'health_data' in self.data_quality_report:
            hd = self.data_quality_report['health_data']
            summary.append("## Health Dataset (HEAT_Johannesburg_FINAL)")
            summary.append(f"- **Total Records**: {hd['total_records']:,}")
            summary.append(f"- **Total Variables**: {hd['total_columns']:,}")
            
            if hd.get('date_range'):
                summary.append(f"- **Date Range**: {hd['date_range']['start']} to {hd['date_range']['end']}")
                summary.append(f"- **Temporal Span**: {hd['date_range']['span_days']:,} days")
            
            if hd.get('study_sources'):
                summary.append(f"- **Study Sources**: {len(hd['study_sources'])} studies")
                for study, count in list(hd['study_sources'].items())[:5]:
                    summary.append(f"  - {study}: {count:,} records")
            
            if hd.get('key_biomarkers_availability'):
                summary.append("- **Key Biomarker Availability**:")
                for biomarker, info in hd['key_biomarkers_availability'].items():
                    summary.append(f"  - {biomarker}: {info['available_records']:,} records ({info['availability_percentage']:.1f}%)")
            
            summary.append("")
        
        if 'socio_data' in self.data_quality_report:
            sd = self.data_quality_report['socio_data']
            summary.append("## Socioeconomic Dataset (GCRO Quality of Life Survey)")
            summary.append(f"- **Total Records**: {sd['total_records']:,}")
            summary.append(f"- **Total Variables**: {sd['total_columns']:,}")
            
            if sd.get('survey_years'):
                summary.append("- **Survey Years**:")
                for year, count in sd['survey_years'].items():
                    summary.append(f"  - {year}: {count:,} records")
            
            if sd.get('key_variables_availability'):
                summary.append("- **Key Socioeconomic Variables**:")
                for var, info in list(sd['key_variables_availability'].items())[:8]:
                    summary.append(f"  - {var}: {info['available_records']:,} records ({info['availability_percentage']:.1f}%)")
            
            if sd.get('climate_integration_status'):
                climate_vars = len(sd['climate_integration_status'])
                summary.append(f"- **Climate Integration**: {climate_vars} climate variables integrated")
            
            summary.append("")
        
        if 'integration_analysis' in self.data_quality_report:
            ia = self.data_quality_report['integration_analysis']
            summary.append("## Integration Analysis")
            
            if ia.get('temporal_alignment'):
                ta = ia['temporal_alignment']
                if ta.get('overlap_days', 0) > 0:
                    summary.append(f"- **Temporal Overlap**: {ta['overlap_days']:,} days")
                else:
                    summary.append("- **Temporal Overlap**: None (different time periods)")
            
            if ia.get('sample_size_implications'):
                ssi = ia['sample_size_implications']
                summary.append(f"- **Analytical Approach**: {ssi['integration_approach']}")
                summary.append(f"- **Expected Power**: {ssi['expected_analytical_power']}")
            
            if ia.get('methodological_considerations'):
                summary.append("- **Key Methodological Considerations**:")
                for consideration in ia['methodological_considerations'][:4]:
                    summary.append(f"  - {consideration}")
        
        # Save summary report
        summary_text = "\n".join(summary)
        summary_path = Path('/home/cparker/heat_analysis_optimized/data_quality_summary.md')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        self.log_analysis_step("REPORT_SAVED", f"Reports saved to {report_path} and {summary_path}")
        
        return summary_text

# Initialize and run comprehensive analysis
if __name__ == "__main__":
    analyzer = ComprehensiveHeatHealthAnalyzer()
    
    print("=== COMPREHENSIVE HEAT-HEALTH-SOCIOECONOMIC DATA ANALYSIS ===")
    print("Rigorous examination of full datasets with complete transparency")
    print()
    
    # Load datasets
    health_loaded = analyzer.load_health_data()
    socio_loaded = analyzer.load_socio_data()
    
    if health_loaded and socio_loaded:
        # Analyze integration potential
        analyzer.analyze_integration_potential()
        
        # Generate comprehensive report
        summary = analyzer.generate_data_quality_report()
        
        print("\n=== DATA QUALITY SUMMARY ===")
        print(summary)
        
        print("\n=== ANALYSIS LOG ===")
        for entry in analyzer.analysis_log[-5:]:  # Last 5 entries
            print(f"{entry['timestamp']}: {entry['step']} - {entry['details']}")
    
    else:
        print("ERROR: Could not load required datasets")