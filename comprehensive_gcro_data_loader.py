#!/usr/bin/env python3
"""
Comprehensive GCRO Data Loader
Loads and harmonizes all 6 GCRO Quality of Life survey waves (2009-2021)

This module handles the complete 119,983 record GCRO dataset spanning 12 years
of socioeconomic and quality of life data from Johannesburg metropolitan area.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveGCROLoader:
    """
    Comprehensive loader for all GCRO Quality of Life survey waves.
    
    Handles data harmonization across different survey years with varying
    variable names, scales, and structures.
    """
    
    def __init__(self, base_path="data/socioeconomic/GCRO/quailty_of_life"):
        self.base_path = Path(base_path)
        self.survey_waves = {
            '2009': {
                'path': '2009/native/csv/gcro-qlf-2009-v1.1-20011209-s10.csv',
                'expected_records': 6637,
                'year': 2009
            },
            '2011': {
                'path': '2011/native/csv/gcro 2011_28feb12nogis-v1-s10.csv',
                'expected_records': 16730,
                'year': 2011
            },
            '2013-2014': {
                'path': '2013-2014/native/qols-iii-2013-2014-v1-csv/qol-iii-2013-2014-v1.csv',
                'expected_records': 27491,
                'year': 2014
            },
            '2015-2016': {
                'path': '2015-2016/native/qol-iv-2015-2016-v1-csv/qol-iv-2014-2015-v1.csv',
                'expected_records': 30618,
                'year': 2016
            },
            '2017-2018': {
                'path': '2017-2018/native/gcro-qols-v-v1.1/qols-v-2017-2018-v1.1.csv',
                'expected_records': 24890,
                'year': 2018
            },
            '2020-2021': {
                'path': '2020-2021/native/qols-2020-2021-v1/qols-2020-2021-new-weights-v1.csv',
                'expected_records': 13617,
                'year': 2021
            }
        }
        
        self.loaded_surveys = {}
        self.harmonized_data = None
        self.data_quality_report = {}
        
        # Common harmonized variables across all waves
        self.core_variables = {
            # Demographics
            'age': ['age', 'q_age', 'resp_age', 'respondent_age'],
            'sex': ['sex', 'gender', 'q_sex', 'resp_sex'],
            'race': ['race', 'population_group', 'q_race', 'ethnic_group'],
            'education': ['education', 'q_education', 'highest_education', 'educ_level'],
            
            # Socioeconomic
            'income': ['income', 'household_income', 'q_income', 'hh_income'],
            'employment': ['employment', 'employment_status', 'q_employment', 'work_status'],
            
            # Housing
            'dwelling_type': ['dwelling_type', 'house_type', 'q_dwelling', 'home_type'],
            'tenure': ['tenure', 'housing_tenure', 'q_tenure', 'ownership'],
            
            # Geographic
            'ward': ['ward', 'ward_code', 'q_ward', 'municipal_ward'],
            'municipality': ['municipality', 'local_municipality', 'q_municipality'],
            
            # Health
            'health_status': ['health', 'health_status', 'q_health', 'self_reported_health'],
            'chronic_conditions': ['chronic_illness', 'chronic_conditions', 'q_chronic'],
            
            # Quality of Life
            'life_satisfaction': ['life_satisfaction', 'q_life_satisfaction', 'satisfaction'],
            'safety': ['safety', 'feel_safe', 'q_safety', 'crime_concern']
        }
        
    def load_survey_wave(self, wave_name):
        """Load a specific survey wave."""
        if wave_name not in self.survey_waves:
            raise ValueError(f"Unknown survey wave: {wave_name}")
        
        wave_info = self.survey_waves[wave_name]
        file_path = self.base_path / wave_info['path']
        
        print(f"📊 Loading {wave_name} GCRO survey...")
        print(f"   📁 Path: {file_path}")
        
        if not file_path.exists():
            print(f"   ❌ File not found: {file_path}")
            return None
        
        try:
            # Load with flexible encoding
            try:
                data = pd.read_csv(file_path, low_memory=False)
            except UnicodeDecodeError:
                data = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
            
            print(f"   ✅ Loaded: {len(data):,} records, {len(data.columns):,} columns")
            
            # Add metadata
            data['survey_wave'] = wave_name
            data['survey_year'] = wave_info['year']
            data['data_source'] = 'GCRO_QoL'
            
            # Validate record count
            expected = wave_info['expected_records']
            actual = len(data)
            if abs(actual - expected) > 100:  # Allow some tolerance
                print(f"   ⚠️  Record count mismatch: expected ~{expected:,}, got {actual:,}")
            
            self.loaded_surveys[wave_name] = data
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading {wave_name}: {str(e)}")
            return None
    
    def load_all_surveys(self):
        """Load all available survey waves."""
        print("=== LOADING ALL GCRO SURVEY WAVES ===")
        
        total_records = 0
        successful_loads = 0
        
        for wave_name in self.survey_waves.keys():
            data = self.load_survey_wave(wave_name)
            if data is not None:
                total_records += len(data)
                successful_loads += 1
        
        print(f"\n📈 Loading Summary:")
        print(f"   ✅ Successfully loaded: {successful_loads}/{len(self.survey_waves)} waves")
        print(f"   📊 Total records: {total_records:,}")
        print(f"   🗓️  Time span: 2009-2021 (12 years)")
        
        return self.loaded_surveys
    
    def harmonize_variable(self, data, target_var, possible_columns):
        """Harmonize a variable across different column names."""
        for col in possible_columns:
            if col in data.columns:
                return data[col]
        return pd.Series([np.nan] * len(data), name=target_var)
    
    def standardize_categorical(self, series, category_type):
        """Standardize categorical variables across survey waves."""
        if category_type == 'sex':
            # Standardize sex/gender
            mapping = {
                'Male': 'Male', 'male': 'Male', 'M': 'Male', '1': 'Male',
                'Female': 'Female', 'female': 'Female', 'F': 'Female', '2': 'Female'
            }
            return series.map(mapping).fillna(series)
        
        elif category_type == 'race':
            # Standardize race/population group
            mapping = {
                'African': 'African', 'Black African': 'African', 'Black': 'African',
                'Coloured': 'Coloured', 'Mixed': 'Coloured',
                'Indian': 'Indian', 'Indian/Asian': 'Indian',
                'White': 'White', 'European': 'White'
            }
            return series.map(mapping).fillna(series)
        
        elif category_type == 'education':
            # Standardize education levels
            mapping = {
                'No schooling': 'No formal education',
                'Some primary': 'Primary incomplete',
                'Primary complete': 'Primary complete', 
                'Some secondary': 'Secondary incomplete',
                'Secondary complete': 'Secondary complete',
                'Matric': 'Secondary complete',
                'Some tertiary': 'Tertiary incomplete',
                'Tertiary complete': 'Tertiary complete',
                'Postgraduate': 'Postgraduate'
            }
            return series.map(mapping).fillna(series)
        
        return series
    
    def harmonize_all_data(self):
        """Harmonize all loaded survey data."""
        if not self.loaded_surveys:
            print("❌ No survey data loaded. Run load_all_surveys() first.")
            return None
        
        print("\n=== HARMONIZING GCRO DATA ACROSS ALL WAVES ===")
        
        harmonized_surveys = []
        
        for wave_name, data in self.loaded_surveys.items():
            print(f"\n🔄 Harmonizing {wave_name}...")
            
            harmonized = pd.DataFrame()
            
            # Add metadata
            harmonized['survey_wave'] = data['survey_wave']
            harmonized['survey_year'] = data['survey_year']
            harmonized['data_source'] = data['data_source']
            
            # Generate unique ID
            harmonized['gcro_id'] = f"{wave_name}_" + (data.index + 1).astype(str)
            
            # Harmonize core variables
            for var_name, possible_cols in self.core_variables.items():
                harmonized_var = self.harmonize_variable(data, var_name, possible_cols)
                
                # Apply standardization based on variable type
                if var_name in ['sex', 'race', 'education']:
                    harmonized_var = self.standardize_categorical(harmonized_var, var_name)
                
                harmonized[var_name] = harmonized_var
                
                # Report harmonization success
                found_cols = [col for col in possible_cols if col in data.columns]
                if found_cols:
                    completeness = harmonized[var_name].notna().mean() * 100
                    print(f"   ✅ {var_name}: {found_cols[0]} ({completeness:.1f}% complete)")
                else:
                    print(f"   ❌ {var_name}: No matching column found")
            
            # Add geographic information if available
            if 'latitude' in data.columns:
                harmonized['latitude'] = data['latitude']
            if 'longitude' in data.columns:
                harmonized['longitude'] = data['longitude']
            
            # Add wave-specific sample weights if available
            weight_cols = ['weight', 'sample_weight', 'wgt', 'weights']
            for weight_col in weight_cols:
                if weight_col in data.columns:
                    harmonized['sample_weight'] = data[weight_col]
                    break
            
            print(f"   📊 Harmonized: {len(harmonized):,} records")
            harmonized_surveys.append(harmonized)
        
        # Combine all waves
        print(f"\n🔗 Combining all survey waves...")
        self.harmonized_data = pd.concat(harmonized_surveys, ignore_index=True)
        
        print(f"✅ HARMONIZATION COMPLETE:")
        print(f"   📊 Total records: {len(self.harmonized_data):,}")
        print(f"   📋 Total variables: {len(self.harmonized_data.columns):,}")
        print(f"   🗓️  Time span: {self.harmonized_data['survey_year'].min()}-{self.harmonized_data['survey_year'].max()}")
        
        return self.harmonized_data
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report."""
        if self.harmonized_data is None:
            print("❌ No harmonized data available. Run harmonize_all_data() first.")
            return None
        
        print("\n=== GENERATING DATA QUALITY REPORT ===")
        
        data = self.harmonized_data
        
        # Overall statistics
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'total_variables': len(data.columns),
            'survey_waves': len(data['survey_wave'].unique()),
            'temporal_span': {
                'start_year': int(data['survey_year'].min()),
                'end_year': int(data['survey_year'].max()),
                'total_years': int(data['survey_year'].max() - data['survey_year'].min())
            }
        }
        
        # Wave-specific statistics
        report['wave_statistics'] = {}
        for wave in data['survey_wave'].unique():
            wave_data = data[data['survey_wave'] == wave]
            report['wave_statistics'][wave] = {
                'records': len(wave_data),
                'year': int(wave_data['survey_year'].iloc[0]),
                'completeness_rate': (wave_data.notna().sum().sum() / (len(wave_data) * len(wave_data.columns))) * 100
            }
        
        # Variable completeness
        report['variable_completeness'] = {}
        for var in self.core_variables.keys():
            if var in data.columns:
                completeness = data[var].notna().mean() * 100
                report['variable_completeness'][var] = {
                    'completeness_rate': float(completeness),
                    'missing_records': int(data[var].isna().sum()),
                    'unique_values': int(data[var].nunique())
                }
        
        # Geographic coverage
        if 'latitude' in data.columns and 'longitude' in data.columns:
            geo_data = data.dropna(subset=['latitude', 'longitude'])
            report['geographic_coverage'] = {
                'records_with_coordinates': len(geo_data),
                'percentage_geocoded': (len(geo_data) / len(data)) * 100,
                'latitude_range': [float(geo_data['latitude'].min()), float(geo_data['latitude'].max())],
                'longitude_range': [float(geo_data['longitude'].min()), float(geo_data['longitude'].max())]
            }
        
        self.data_quality_report = report
        
        # Save report
        report_path = Path('validation/gcro_comprehensive_data_quality_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Data quality report saved: {report_path}")
        
        # Print summary
        print(f"\n📊 GCRO DATA QUALITY SUMMARY:")
        print(f"   • Total records: {report['total_records']:,}")
        print(f"   • Survey waves: {report['survey_waves']}")
        print(f"   • Time span: {report['temporal_span']['start_year']}-{report['temporal_span']['end_year']}")
        print(f"   • Variables harmonized: {len([v for v in report['variable_completeness'] if report['variable_completeness'][v]['completeness_rate'] > 0])}")
        
        return report
    
    def save_harmonized_data(self, output_path="data/gcro_harmonized_complete.csv"):
        """Save the harmonized dataset."""
        if self.harmonized_data is None:
            print("❌ No harmonized data to save. Run harmonize_all_data() first.")
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving harmonized GCRO dataset...")
        self.harmonized_data.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024*1024)  # MB
        
        print(f"✅ Saved: {output_path}")
        print(f"   📊 Records: {len(self.harmonized_data):,}")
        print(f"   📋 Variables: {len(self.harmonized_data.columns):,}")
        print(f"   💾 File size: {file_size:.1f} MB")
        
        return True
    
    def get_summary_statistics(self):
        """Get summary statistics for the harmonized dataset."""
        if self.harmonized_data is None:
            return None
        
        data = self.harmonized_data
        
        print(f"\n=== GCRO COMPREHENSIVE DATASET SUMMARY ===")
        print(f"📊 Scale: {len(data):,} records across {len(data['survey_wave'].unique())} waves")
        print(f"🗓️  Period: {data['survey_year'].min()}-{data['survey_year'].max()}")
        print(f"📍 Coverage: Johannesburg metropolitan area")
        
        print(f"\n📈 Records by Survey Wave:")
        wave_counts = data['survey_wave'].value_counts().sort_index()
        for wave, count in wave_counts.items():
            print(f"   • {wave}: {count:,} records")
        
        print(f"\n🎯 Key Variable Completeness:")
        key_vars = ['age', 'sex', 'race', 'education', 'income']
        for var in key_vars:
            if var in data.columns:
                completeness = data[var].notna().mean() * 100
                print(f"   • {var.capitalize()}: {completeness:.1f}%")
        
        return wave_counts

def main():
    """Main function to demonstrate the comprehensive GCRO data loader."""
    print("🚀 COMPREHENSIVE GCRO DATA LOADER")
    print("Loading all 6 survey waves (2009-2021) with ~120k records")
    print("=" * 60)
    
    # Initialize loader
    loader = ComprehensiveGCROLoader()
    
    # Load all survey waves
    surveys = loader.load_all_surveys()
    
    if not surveys:
        print("❌ Failed to load any survey data")
        return False
    
    # Harmonize data across waves  
    harmonized = loader.harmonize_all_data()
    
    if harmonized is None:
        print("❌ Failed to harmonize data")
        return False
    
    # Generate quality report
    quality_report = loader.generate_data_quality_report()
    
    # Save harmonized dataset
    success = loader.save_harmonized_data()
    
    # Print summary
    loader.get_summary_statistics()
    
    print(f"\n🎉 SUCCESS: Comprehensive GCRO dataset ready for analysis!")
    print(f"   📊 {len(harmonized):,} records harmonized across 12 years")
    print(f"   📁 Saved to: data/gcro_harmonized_complete.csv")
    print(f"   📋 Quality report: validation/gcro_comprehensive_data_quality_report.json")
    
    return success

if __name__ == "__main__":
    main()