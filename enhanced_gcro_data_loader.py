#!/usr/bin/env python3
"""
Enhanced GCRO Data Loader with Correct Variable Mappings
Uses actual metadata to properly harmonize all 6 survey waves (2009-2021)

This version uses the correct variable codes from GCRO documentation to 
properly extract demographics, socioeconomics, and health variables.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedGCROLoader:
    """
    Enhanced loader with correct variable mappings from GCRO metadata.
    """
    
    def __init__(self, base_path="data/socioeconomic/GCRO/quailty_of_life"):
        self.base_path = Path(base_path)
        
        # Survey wave configurations with correct variable mappings
        self.survey_waves = {
            '2009': {
                'path': '2009/native/csv/gcro-qlf-2009-v1.1-20011209-s10.csv',
                'expected_records': 6637,
                'year': 2009,
                'encoding': 'latin-1',
                'variables': {
                    'age': 'A_12_2',
                    'sex': 'A2', 
                    'race': 'A1',
                    'education': 'A_12_1',
                    'income': 'A_12_7',
                    'employment': 'A_8_6',
                    'dwelling_type': 'A3',
                    'ward': 'Ward'
                }
            },
            '2011': {
                'path': '2011/native/csv/gcro 2011_28feb12nogis-v1-s10.csv',
                'expected_records': 16730,
                'year': 2011,
                'encoding': 'latin-1',
                'variables': {
                    'age': 'A_12_2',
                    'sex': 'A2',
                    'race': 'A1', 
                    'education': 'A_12_1',
                    'income': 'A_12_7',
                    'employment': 'A_8_6',
                    'dwelling_type': 'A3',
                    'ward': 'Ward'
                }
            },
            '2013-2014': {
                'path': '2013-2014/native/qols-iii-2013-2014-v1-csv/qol-iii-2013-2014-v1.csv',
                'expected_records': 27491,
                'year': 2014,
                'encoding': 'utf-8',
                'variables': {
                    'age': 'age',
                    'sex': 'sex',
                    'race': 'population_group',
                    'education': 'education',
                    'income': 'household_income',
                    'employment': 'employment_status',
                    'dwelling_type': 'dwelling_type',
                    'ward': 'ward'
                }
            },
            '2015-2016': {
                'path': '2015-2016/native/qol-iv-2015-2016-v1-csv/qol-iv-2014-2015-v1.csv',
                'expected_records': 30618,
                'year': 2016,
                'encoding': 'utf-8',
                'variables': {
                    'age': 'age',
                    'sex': 'sex',
                    'race': 'population_group',
                    'education': 'education',
                    'income': 'household_income', 
                    'employment': 'employment_status',
                    'dwelling_type': 'dwelling_type',
                    'ward': 'ward'
                }
            },
            '2017-2018': {
                'path': '2017-2018/native/gcro-qols-v-v1.1/qols-v-2017-2018-v1.1.csv',
                'expected_records': 24890,
                'year': 2018,
                'encoding': 'utf-8',
                'variables': {
                    'age': 'age',
                    'sex': 'sex',
                    'race': 'population_group',
                    'education': 'education',
                    'income': 'household_income',
                    'employment': 'employment_status',
                    'dwelling_type': 'dwelling_type',
                    'ward': 'ward'
                }
            },
            '2020-2021': {
                'path': '2020-2021/native/qols-2020-2021-v1/qols-2020-2021-new-weights-v1.csv',
                'expected_records': 13617,
                'year': 2021,
                'encoding': 'utf-8',
                'variables': {
                    'age': 'age',
                    'sex': 'sex',
                    'race': 'population_group',
                    'education': 'education',
                    'income': 'household_income',
                    'employment': 'employment_status',
                    'dwelling_type': 'dwelling_type',
                    'ward': 'ward_code'
                }
            }
        }
        
        self.loaded_surveys = {}
        self.harmonized_data = None
        
    def load_survey_wave(self, wave_name):
        """Load a specific survey wave with correct encoding."""
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
            # Load with correct encoding
            data = pd.read_csv(file_path, low_memory=False, encoding=wave_info['encoding'])
            
            print(f"   ✅ Loaded: {len(data):,} records, {len(data.columns):,} columns")
            
            # Add metadata
            data['survey_wave'] = wave_name
            data['survey_year'] = wave_info['year']
            data['data_source'] = 'GCRO_QoL'
            
            self.loaded_surveys[wave_name] = {
                'data': data,
                'variables': wave_info['variables']
            }
            
            return data
            
        except Exception as e:
            print(f"   ❌ Error loading {wave_name}: {str(e)}")
            return None
    
    def standardize_categorical(self, series, category_type, wave_name):
        """Standardize categorical variables with wave-specific mappings."""
        if category_type == 'sex':
            if wave_name in ['2009', '2011']:
                # Coded values for older surveys
                mapping = {1: 'Male', 2: 'Female', '1': 'Male', '2': 'Female',
                          'Male': 'Male', 'Female': 'Female'}
            else:
                # Text values for newer surveys
                mapping = {'Male': 'Male', 'Female': 'Female', 'male': 'Male', 'female': 'Female'}
            return series.map(mapping).fillna(series)
        
        elif category_type == 'race':
            # Standardize across different codings
            mapping = {
                1: 'African', 2: 'Coloured', 3: 'Indian', 4: 'White',
                '1': 'African', '2': 'Coloured', '3': 'Indian', '4': 'White',
                'African': 'African', 'Black African': 'African', 'Black': 'African',
                'Coloured': 'Coloured', 'Mixed': 'Coloured',
                'Indian': 'Indian', 'Indian/Asian': 'Indian', 'Asian': 'Indian',
                'White': 'White'
            }
            return series.map(mapping).fillna(series)
        
        elif category_type == 'education':
            # Standardize education levels
            mapping = {
                1: 'No formal education', 2: 'Primary incomplete', 3: 'Primary complete',
                4: 'Secondary incomplete', 5: 'Secondary complete', 6: 'Tertiary incomplete',
                7: 'Tertiary complete', 8: 'Postgraduate',
                '1': 'No formal education', '2': 'Primary incomplete', '3': 'Primary complete',
                '4': 'Secondary incomplete', '5': 'Secondary complete', '6': 'Tertiary incomplete', 
                '7': 'Tertiary complete', '8': 'Postgraduate',
                'No schooling': 'No formal education',
                'Some primary': 'Primary incomplete',
                'Primary complete': 'Primary complete',
                'Some secondary': 'Secondary incomplete', 
                'Matric': 'Secondary complete',
                'Some tertiary': 'Tertiary incomplete',
                'Tertiary complete': 'Tertiary complete'
            }
            return series.map(mapping).fillna(series)
        
        return series
    
    def harmonize_all_data(self):
        """Harmonize all loaded survey data with correct variable mappings."""
        if not self.loaded_surveys:
            print("❌ No survey data loaded. Run load_all_surveys() first.")
            return None
        
        print("\n=== HARMONIZING GCRO DATA WITH CORRECT MAPPINGS ===")
        
        harmonized_surveys = []
        
        for wave_name, wave_data in self.loaded_surveys.items():
            data = wave_data['data']
            var_mapping = wave_data['variables']
            
            print(f"\n🔄 Harmonizing {wave_name}...")
            
            harmonized = pd.DataFrame()
            
            # Add metadata
            harmonized['gcro_id'] = f"{wave_name}_" + (data.index + 1).astype(str)
            harmonized['survey_wave'] = data['survey_wave']
            harmonized['survey_year'] = data['survey_year']
            harmonized['data_source'] = data['data_source']
            
            # Harmonize each variable using correct mappings
            for target_var, source_var in var_mapping.items():
                if source_var in data.columns:
                    harmonized_var = data[source_var].copy()
                    
                    # Apply standardization
                    if target_var in ['sex', 'race', 'education']:
                        harmonized_var = self.standardize_categorical(harmonized_var, target_var, wave_name)
                    
                    # Handle numeric variables
                    elif target_var in ['age', 'income']:
                        harmonized_var = pd.to_numeric(harmonized_var, errors='coerce')
                    
                    harmonized[target_var] = harmonized_var
                    
                    # Report success
                    completeness = harmonized[target_var].notna().mean() * 100
                    print(f"   ✅ {target_var}: {source_var} ({completeness:.1f}% complete)")
                else:
                    # Variable not found
                    harmonized[target_var] = np.nan
                    print(f"   ❌ {target_var}: {source_var} not found")
            
            print(f"   📊 Harmonized: {len(harmonized):,} records")
            harmonized_surveys.append(harmonized)
        
        # Combine all waves
        print(f"\n🔗 Combining all survey waves...")
        self.harmonized_data = pd.concat(harmonized_surveys, ignore_index=True)
        
        print(f"✅ ENHANCED HARMONIZATION COMPLETE:")
        print(f"   📊 Total records: {len(self.harmonized_data):,}")
        print(f"   📋 Total variables: {len(self.harmonized_data.columns):,}")
        print(f"   🗓️  Time span: {self.harmonized_data['survey_year'].min()}-{self.harmonized_data['survey_year'].max()}")
        
        return self.harmonized_data
    
    def load_all_surveys(self):
        """Load all survey waves."""
        print("=== LOADING ALL GCRO SURVEY WAVES (ENHANCED) ===")
        
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
        
        return self.loaded_surveys
    
    def generate_enhanced_quality_report(self):
        """Generate enhanced data quality report."""
        if self.harmonized_data is None:
            return None
        
        print("\n=== GENERATING ENHANCED DATA QUALITY REPORT ===")
        
        data = self.harmonized_data
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'survey_waves': len(data['survey_wave'].unique()),
            'temporal_span': f"{data['survey_year'].min()}-{data['survey_year'].max()}",
            'wave_statistics': {},
            'variable_completeness': {},
            'demographic_distribution': {}
        }
        
        # Wave statistics
        for wave in data['survey_wave'].unique():
            wave_data = data[data['survey_wave'] == wave]
            report['wave_statistics'][wave] = {
                'records': len(wave_data),
                'year': int(wave_data['survey_year'].iloc[0])
            }
        
        # Variable completeness
        core_vars = ['age', 'sex', 'race', 'education', 'income', 'employment']
        for var in core_vars:
            if var in data.columns:
                completeness = data[var].notna().mean() * 100
                report['variable_completeness'][var] = {
                    'completeness_rate': float(completeness),
                    'valid_records': int(data[var].notna().sum())
                }
        
        # Demographic distribution
        if 'sex' in data.columns:
            sex_dist = data['sex'].value_counts(dropna=False)
            report['demographic_distribution']['sex'] = sex_dist.to_dict()
        
        if 'race' in data.columns:
            race_dist = data['race'].value_counts(dropna=False) 
            report['demographic_distribution']['race'] = race_dist.to_dict()
        
        # Save report
        report_path = Path('validation/enhanced_gcro_data_quality_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Enhanced quality report saved: {report_path}")
        return report
    
    def save_enhanced_data(self, output_path="data/gcro_enhanced_harmonized.csv"):
        """Save the enhanced harmonized dataset."""
        if self.harmonized_data is None:
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving enhanced GCRO dataset...")
        self.harmonized_data.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024*1024)  # MB
        
        print(f"✅ Saved: {output_path}")
        print(f"   📊 Records: {len(self.harmonized_data):,}")
        print(f"   📋 Variables: {len(self.harmonized_data.columns):,}")  
        print(f"   💾 File size: {file_size:.1f} MB")
        
        return True
    
    def get_data_summary(self):
        """Print comprehensive data summary."""
        if self.harmonized_data is None:
            return None
        
        data = self.harmonized_data
        
        print(f"\n=== ENHANCED GCRO DATASET SUMMARY ===")
        print(f"📊 Total Records: {len(data):,}")
        print(f"🗓️  Period: {data['survey_year'].min()}-{data['survey_year'].max()}")
        print(f"📍 Geographic Area: Johannesburg metropolitan area")
        
        print(f"\n📈 Records by Survey Wave:")
        for wave in sorted(data['survey_wave'].unique()):
            count = len(data[data['survey_wave'] == wave])
            year = data[data['survey_wave'] == wave]['survey_year'].iloc[0]
            print(f"   • {wave} ({year}): {count:,} records")
        
        print(f"\n🎯 Variable Completeness:")
        core_vars = ['age', 'sex', 'race', 'education', 'income', 'employment']
        for var in core_vars:
            if var in data.columns:
                completeness = data[var].notna().mean() * 100
                valid_count = data[var].notna().sum()
                print(f"   • {var.capitalize()}: {completeness:.1f}% ({valid_count:,} records)")
        
        print(f"\n👥 Sample Demographics (where available):")
        if 'sex' in data.columns:
            sex_counts = data['sex'].value_counts(dropna=False)
            print(f"   • Sex distribution: {dict(sex_counts)}")
        
        if 'age' in data.columns:
            age_stats = data['age'].describe()
            print(f"   • Age: Mean={age_stats['mean']:.1f}, Median={age_stats['50%']:.1f}, Range={age_stats['min']:.0f}-{age_stats['max']:.0f}")

def main():
    """Main function for enhanced GCRO data processing."""
    print("🚀 ENHANCED GCRO DATA LOADER WITH CORRECT MAPPINGS")
    print("Processing 119k+ records across 6 survey waves (2009-2021)")
    print("=" * 60)
    
    # Initialize enhanced loader
    loader = EnhancedGCROLoader()
    
    # Load all surveys
    surveys = loader.load_all_surveys()
    if not surveys:
        return False
    
    # Harmonize with correct mappings
    harmonized = loader.harmonize_all_data()
    if harmonized is None:
        return False
    
    # Generate enhanced quality report
    quality_report = loader.generate_enhanced_quality_report()
    
    # Save enhanced dataset
    success = loader.save_enhanced_data()
    
    # Print comprehensive summary
    loader.get_data_summary()
    
    print(f"\n🎉 SUCCESS: Enhanced GCRO dataset ready!")
    print(f"   📊 {len(harmonized):,} records properly harmonized")
    print(f"   📁 Saved to: data/gcro_enhanced_harmonized.csv")
    print(f"   📋 Quality report: validation/enhanced_gcro_data_quality_report.json")
    
    return success

if __name__ == "__main__":
    main()