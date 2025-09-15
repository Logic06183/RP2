#!/usr/bin/env python3

"""
Publication-Ready Descriptive Statistics Tables Generator
========================================================

Creates comprehensive descriptive statistics tables for the climate-health
analysis publication, formatted for scientific journals.

Tables generated:
1. Table 1: Study Population Characteristics
2. Table 2: Climate Exposure Summary Statistics  
3. Table 3: DLNM Model Results Summary
4. Table 4: Temperature-Health Association Results

Author: Claude Code - Heat-Health Research Team
Date: 2025-09-05
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class PublicationTablesGenerator:
    """Generate publication-ready tables for climate-health analysis"""
    
    def __init__(self, gcro_path, dlnm_results_path, enhanced_results_path):
        self.gcro_path = gcro_path
        self.dlnm_results_path = dlnm_results_path
        self.enhanced_results_path = enhanced_results_path
        self.load_data()
    
    def load_data(self):
        """Load all necessary data sources"""
        print("📂 Loading data for publication tables...")
        
        # Load GCRO socioeconomic data
        self.gcro_data = pd.read_csv(self.gcro_path)
        
        # Load DLNM analysis results
        with open(self.dlnm_results_path, 'r') as f:
            self.dlnm_results = json.load(f)
        
        # Load enhanced analysis results
        with open(self.enhanced_results_path, 'r') as f:
            self.enhanced_results = json.load(f)
        
        # Prepare analysis data
        self._prepare_analysis_data()
        
        print(f"   • Data loaded successfully: {len(self.analysis_data):,} observations")
    
    def _prepare_analysis_data(self):
        """Prepare analysis dataset with all variables"""
        
        # Parse dates
        self.gcro_data['date'] = pd.to_datetime(self.gcro_data['interview_date_parsed'])
        self.gcro_data['year'] = self.gcro_data['date'].dt.year
        
        # Primary temperature variable
        self.gcro_data['temperature'] = self.gcro_data['era5_temp_30d_mean']
        
        # Health outcomes
        cardio_conditions = ['q13_11_6_hypertension', 'q13_11_5_heart']
        self.gcro_data['cardiovascular_score'] = 0
        for condition in cardio_conditions:
            if condition in self.gcro_data.columns:
                self.gcro_data['cardiovascular_score'] += (
                    self.gcro_data[condition] == 'Yes'
                ).astype(int)
        self.gcro_data['cardiovascular_score'] = (
            self.gcro_data['cardiovascular_score'] / len(cardio_conditions)
        )
        
        # Renal risk
        self.gcro_data['renal_risk'] = (
            (self.gcro_data['q13_11_2_diabetes'] == 'Yes') &
            (self.gcro_data['q13_6_health_status'].isin(['Poor', 'Fair']))
        ).astype(int)
        
        # Create SES vulnerability index (simplified)
        self.gcro_data['income_numeric'] = pd.Categorical(
            self.gcro_data['q15_3_income'], 
            ordered=True
        ).codes
        self.gcro_data['education_numeric'] = pd.Categorical(
            self.gcro_data['q14_1_education'], 
            ordered=True
        ).codes
        
        # Complete cases
        required_cols = ['temperature', 'cardiovascular_score', 'renal_risk', 'date']
        self.analysis_data = self.gcro_data.dropna(subset=required_cols).copy()
        
        # Create temperature categories
        temp_percentiles = self.analysis_data['temperature'].quantile([0.25, 0.75, 0.90])
        self.analysis_data['temp_category'] = pd.cut(
            self.analysis_data['temperature'],
            bins=[-np.inf, temp_percentiles[0.25], temp_percentiles[0.75], 
                  temp_percentiles[0.90], np.inf],
            labels=['Cool', 'Moderate', 'Warm', 'Hot']
        )
    
    def create_table1_population_characteristics(self):
        """Table 1: Study Population Characteristics"""
        
        print("📊 Creating Table 1: Study Population Characteristics...")
        
        # Calculate demographics
        n_total = len(self.analysis_data)
        
        # Age (extract numeric from categories)
        age_mapping = {
            '18-19': 18.5, '20-24': 22, '25-29': 27, '30-34': 32,
            '35-39': 37, '40-44': 42, '45-49': 47, '50-54': 52,
            '55-59': 57, '60-64': 62, '65+': 70
        }
        self.analysis_data['age_numeric'] = self.analysis_data['q14_2_age_recode'].map(age_mapping)
        age_mean = self.analysis_data['age_numeric'].mean()
        age_std = self.analysis_data['age_numeric'].std()
        
        # Gender
        female_pct = (self.analysis_data['a2_sex'] == 'Female').mean() * 100
        
        # Education
        education_counts = self.analysis_data['q14_1_education'].value_counts()
        education_total = education_counts.sum()
        
        # Income
        income_counts = self.analysis_data['q15_3_income'].value_counts()
        income_total = income_counts.sum()
        
        # Health conditions
        hypertension_pct = (self.analysis_data['q13_11_6_hypertension'] == 'Yes').mean() * 100
        heart_disease_pct = (self.analysis_data['q13_11_5_heart'] == 'Yes').mean() * 100
        diabetes_pct = (self.analysis_data['q13_11_2_diabetes'] == 'Yes').mean() * 100
        
        # Climate exposures
        temp_mean = self.analysis_data['temperature'].mean()
        temp_std = self.analysis_data['temperature'].std()
        temp_range = (self.analysis_data['temperature'].min(), 
                     self.analysis_data['temperature'].max())
        
        # Temperature category distributions
        temp_cat_counts = self.analysis_data['temp_category'].value_counts()
        
        # Build table
        table1_data = []
        
        # Demographics section
        table1_data.append(['Demographics', '', ''])
        table1_data.append(['Total participants', f'{n_total:,}', ''])
        table1_data.append(['Age, years (mean ± SD)', 
                           f'{age_mean:.1f} ± {age_std:.1f}', 
                           f'[{self.analysis_data["age_numeric"].min():.0f}–{self.analysis_data["age_numeric"].max():.0f}]'])
        table1_data.append(['Female sex, n (%)', 
                           f'{int(n_total * female_pct/100):,} ({female_pct:.1f})', ''])
        
        # Education section
        table1_data.append(['Education level, n (%)', '', ''])
        for edu_level, count in education_counts.head(3).items():
            pct = (count / education_total) * 100
            table1_data.append([f'  {edu_level}', f'{count:,} ({pct:.1f})', ''])
        
        # Income section  
        table1_data.append(['Monthly income, n (%)', '', ''])
        for income_level, count in income_counts.head(3).items():
            pct = (count / income_total) * 100
            clean_income = income_level.replace('R', 'ZAR ') if 'R' in str(income_level) else str(income_level)
            table1_data.append([f'  {clean_income}', f'{count:,} ({pct:.1f})', ''])
        
        # Health conditions section
        table1_data.append(['Health conditions, n (%)', '', ''])
        table1_data.append(['Hypertension', 
                           f'{int(n_total * hypertension_pct/100):,} ({hypertension_pct:.1f})', ''])
        table1_data.append(['Heart disease', 
                           f'{int(n_total * heart_disease_pct/100):,} ({heart_disease_pct:.1f})', ''])
        table1_data.append(['Diabetes', 
                           f'{int(n_total * diabetes_pct/100):,} ({diabetes_pct:.1f})', ''])
        
        # Climate exposure section
        table1_data.append(['Climate exposure', '', ''])
        table1_data.append(['Temperature, °C (mean ± SD)', 
                           f'{temp_mean:.1f} ± {temp_std:.1f}', 
                           f'[{temp_range[0]:.1f}–{temp_range[1]:.1f}]'])
        
        # Temperature categories
        for temp_cat, count in temp_cat_counts.items():
            pct = (count / len(self.analysis_data)) * 100
            table1_data.append([f'  {temp_cat} temperature days', 
                               f'{count:,} ({pct:.1f})', ''])
        
        # Study period
        date_range = (self.analysis_data['date'].min().strftime('%b %Y'),
                     self.analysis_data['date'].max().strftime('%b %Y'))
        table1_data.append(['Study period', f'{date_range[0]} – {date_range[1]}', ''])
        
        # Convert to DataFrame
        table1_df = pd.DataFrame(table1_data, 
                                columns=['Characteristic', 'Value', 'Range'])
        
        return table1_df
    
    def create_table2_climate_exposure(self):
        """Table 2: Climate Exposure Summary Statistics"""
        
        print("📊 Creating Table 2: Climate Exposure Summary Statistics...")
        
        # Temperature statistics by source
        temp_sources = {
            'ERA5 Temperature (30-day)': 'era5_temp_30d_mean',
            'ERA5 Land Surface Temp': 'era5_lst_30d_mean', 
            'MODIS Land Surface Temp': 'modis_lst_30d_mean'
        }
        
        table2_data = []
        
        for source_name, col_name in temp_sources.items():
            if col_name in self.analysis_data.columns:
                values = self.analysis_data[col_name].dropna()
                if len(values) > 0:
                    table2_data.append([
                        source_name,
                        len(values),
                        f'{values.mean():.2f}',
                        f'{values.std():.2f}',
                        f'{values.min():.1f}',
                        f'{values.quantile(0.25):.1f}',
                        f'{values.median():.1f}',
                        f'{values.quantile(0.75):.1f}',
                        f'{values.max():.1f}'
                    ])
        
        # Extreme temperature statistics
        temp_90th = self.analysis_data['temperature'].quantile(0.90)
        temp_95th = self.analysis_data['temperature'].quantile(0.95)
        
        extreme_stats = []
        extreme_stats.append([
            'Extreme heat days (≥90th percentile)',
            f'{(self.analysis_data["temperature"] >= temp_90th).sum():,}',
            f'{temp_90th:.1f}°C threshold',
            f'{(self.analysis_data["temperature"] >= temp_90th).mean()*100:.1f}%',
            '', '', '', '', ''
        ])
        
        extreme_stats.append([
            'Very extreme heat days (≥95th percentile)',
            f'{(self.analysis_data["temperature"] >= temp_95th).sum():,}',
            f'{temp_95th:.1f}°C threshold', 
            f'{(self.analysis_data["temperature"] >= temp_95th).mean()*100:.1f}%',
            '', '', '', '', ''
        ])
        
        # Seasonal patterns
        self.analysis_data['season'] = self.analysis_data['date'].dt.month.map({
            12: 'Summer', 1: 'Summer', 2: 'Summer',
            3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
            6: 'Winter', 7: 'Winter', 8: 'Winter',
            9: 'Spring', 10: 'Spring', 11: 'Spring'
        })
        
        seasonal_stats = []
        for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
            season_data = self.analysis_data[self.analysis_data['season'] == season]['temperature']
            if len(season_data) > 0:
                seasonal_stats.append([
                    f'{season} temperature',
                    len(season_data),
                    f'{season_data.mean():.2f}',
                    f'{season_data.std():.2f}',
                    f'{season_data.min():.1f}',
                    f'{season_data.quantile(0.25):.1f}',
                    f'{season_data.median():.1f}',
                    f'{season_data.quantile(0.75):.1f}',
                    f'{season_data.max():.1f}'
                ])
        
        # Combine all data
        all_data = table2_data + [['', '', '', '', '', '', '', '', '']] + extreme_stats + [['', '', '', '', '', '', '', '', '']] + seasonal_stats
        
        table2_df = pd.DataFrame(all_data, columns=[
            'Climate Variable', 'N', 'Mean', 'SD', 'Min', 'Q25', 'Median', 'Q75', 'Max'
        ])
        
        return table2_df
    
    def create_table3_dlnm_results(self):
        """Table 3: DLNM Model Results Summary"""
        
        print("📊 Creating Table 3: DLNM Model Results Summary...")
        
        table3_data = []
        
        pathways = ['cardiovascular', 'renal']
        
        for pathway in pathways:
            if pathway in self.dlnm_results['dlnm_results']:
                pathway_data = self.dlnm_results['dlnm_results'][pathway]
                
                # Get best model results
                best_model = pathway_data.get('best_model', 'elastic_net')
                if best_model in pathway_data:
                    results = pathway_data[best_model]
                    
                    table3_data.append([
                        pathway.title(),
                        best_model.replace('_', ' ').title(),
                        f'{results["cv_mean"]:.4f}',
                        f'{results["cv_std"]:.4f}',
                        f'{results["r2_full"]:.4f}',
                        f'{results["mse_full"]:.4f}',
                        'Significant' if results["cv_mean"] > 0.01 else 'Not significant'
                    ])
        
        # Add temperature response results if available
        if 'temperature_response_analysis' in self.dlnm_results:
            temp_response = self.dlnm_results['temperature_response_analysis']
            
            for pathway in pathways:
                if pathway in temp_response:
                    temp_data = temp_response[pathway]
                    table3_data.append([
                        f'{pathway.title()} - Temperature Response',
                        'Peak Effect Analysis',
                        f'{temp_data["peak_temperature"]:.1f}°C',
                        f'{temp_data["peak_effect"]:.4f}',
                        f'{temp_data["temperature_range"]["min"]:.1f}',
                        f'{temp_data["temperature_range"]["max"]:.1f}',
                        'Temperature-dependent'
                    ])
        
        # Add lag effects results if available
        if 'lag_effects_analysis' in self.dlnm_results:
            lag_effects = self.dlnm_results['lag_effects_analysis']
            
            for pathway in pathways:
                if pathway in lag_effects:
                    lag_data = lag_effects[pathway]
                    table3_data.append([
                        f'{pathway.title()} - Lag Effects',
                        'Distributed Lag Model',
                        f'{lag_data["lag_at_max_effect"]} days',
                        f'{lag_data["max_lag_effect"]:.4f}',
                        f'{lag_data["extreme_temperature"]:.1f}°C',
                        '21 days max',
                        'Lag-dependent'
                    ])
        
        table3_df = pd.DataFrame(table3_data, columns=[
            'Model/Analysis', 'Method', 'Primary Result', 'Uncertainty', 
            'Secondary Result', 'Additional Info', 'Interpretation'
        ])
        
        return table3_df
    
    def create_table4_associations(self):
        """Table 4: Temperature-Health Association Results"""
        
        print("📊 Creating Table 4: Temperature-Health Association Results...")
        
        # Use enhanced analysis results for associations
        enhanced_stats = self.enhanced_results.get('enhanced_statistical_results', {})
        
        table4_data = []
        
        # Temperature correlations
        correlations = {
            'Cardiovascular': 'cardiovascular_temperature_correlation',
            'Renal': 'renal_temperature_correlation'
        }
        
        for pathway, key in correlations.items():
            if key in enhanced_stats:
                stats = enhanced_stats[key]
                table4_data.append([
                    pathway,
                    'Temperature Correlation',
                    f'{stats["correlation"]:.4f}',
                    f'[{stats["confidence_interval"][0]:.4f}, {stats["confidence_interval"][1]:.4f}]',
                    f'{stats["p_value"]:.4f}',
                    'Yes' if stats["significant"] == "True" else 'No',
                    f'{stats["sample_size"]:,}'
                ])
        
        # Extreme heat effects
        extreme_effects = {
            'Cardiovascular': 'cardiovascular_extreme_heat_effect',
            'Renal': 'renal_extreme_heat_effect'
        }
        
        for pathway, key in extreme_effects.items():
            if key in enhanced_stats:
                stats = enhanced_stats[key]
                table4_data.append([
                    pathway,
                    'Extreme Heat Effect',
                    f'{stats["mean_difference"]:.2f}',
                    f'Cohen\'s d = {stats["cohens_d"]:.3f}',
                    f'{stats["p_value"]:.4f}',
                    'Yes' if stats["significant"] == "True" else 'No',
                    f'{stats["extreme_n"]} vs {stats["normal_n"]}'
                ])
        
        # Temperature thresholds analysis
        temp_thresholds = [
            ('75th percentile', self.analysis_data['temperature'].quantile(0.75)),
            ('90th percentile', self.analysis_data['temperature'].quantile(0.90)),
            ('95th percentile', self.analysis_data['temperature'].quantile(0.95))
        ]
        
        for threshold_name, threshold_temp in temp_thresholds:
            high_temp_mask = self.analysis_data['temperature'] >= threshold_temp
            
            # Cardiovascular effects above threshold
            cardio_high = self.analysis_data[high_temp_mask]['cardiovascular_score'].mean()
            cardio_normal = self.analysis_data[~high_temp_mask]['cardiovascular_score'].mean()
            cardio_diff = cardio_high - cardio_normal
            
            # Renal effects above threshold
            renal_high = self.analysis_data[high_temp_mask]['renal_risk'].mean()
            renal_normal = self.analysis_data[~high_temp_mask]['renal_risk'].mean() 
            renal_diff = renal_high - renal_normal
            
            table4_data.append([
                'Cardiovascular',
                f'{threshold_name} (≥{threshold_temp:.1f}°C)',
                f'{cardio_diff:.4f}',
                f'{cardio_high:.4f} vs {cardio_normal:.4f}',
                'N/A',
                'Descriptive',
                f'{high_temp_mask.sum()} high temp days'
            ])
            
            table4_data.append([
                'Renal',
                f'{threshold_name} (≥{threshold_temp:.1f}°C)',
                f'{renal_diff:.4f}',
                f'{renal_high:.4f} vs {renal_normal:.4f}',
                'N/A', 
                'Descriptive',
                f'{high_temp_mask.sum()} high temp days'
            ])
        
        table4_df = pd.DataFrame(table4_data, columns=[
            'Pathway', 'Analysis', 'Effect Size', 'Details', 'P-value', 
            'Significant', 'Sample Info'
        ])
        
        return table4_df
    
    def save_all_tables(self, output_dir):
        """Generate and save all publication tables"""
        
        print("📋 Generating all publication tables...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all tables
        table1 = self.create_table1_population_characteristics()
        table2 = self.create_table2_climate_exposure()
        table3 = self.create_table3_dlnm_results()
        table4 = self.create_table4_associations()
        
        # Save as CSV files
        table1.to_csv(os.path.join(output_dir, 'Table1_Population_Characteristics.csv'), index=False)
        table2.to_csv(os.path.join(output_dir, 'Table2_Climate_Exposure_Statistics.csv'), index=False)
        table3.to_csv(os.path.join(output_dir, 'Table3_DLNM_Model_Results.csv'), index=False)
        table4.to_csv(os.path.join(output_dir, 'Table4_Temperature_Health_Associations.csv'), index=False)
        
        # Create formatted versions for publication
        self._create_formatted_tables(output_dir, table1, table2, table3, table4)
        
        print(f"✅ All tables saved to {output_dir}")
        
        return {
            'table1': table1,
            'table2': table2, 
            'table3': table3,
            'table4': table4
        }
    
    def _create_formatted_tables(self, output_dir, table1, table2, table3, table4):
        """Create publication-formatted versions of tables"""
        
        # Create a formatted text version suitable for manuscript
        with open(os.path.join(output_dir, 'Publication_Tables_Formatted.txt'), 'w') as f:
            f.write("PUBLICATION TABLES - DLNM CLIMATE-HEALTH ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Table 1
            f.write("TABLE 1. Study Population Characteristics\n")
            f.write("-" * 50 + "\n")
            for _, row in table1.iterrows():
                f.write(f"{row['Characteristic']:<40} {row['Value']:<20} {row['Range']}\n")
            f.write("\n\n")
            
            # Table 2
            f.write("TABLE 2. Climate Exposure Summary Statistics\n")
            f.write("-" * 50 + "\n")
            for _, row in table2.iterrows():
                if row['Climate Variable']:  # Skip empty rows
                    f.write(f"{row['Climate Variable']:<30} N={row['N']:<6} Mean={row['Mean']:<8} SD={row['SD']:<8}\n")
            f.write("\n\n")
            
            # Table 3
            f.write("TABLE 3. DLNM Model Results Summary\n")
            f.write("-" * 50 + "\n")
            for _, row in table3.iterrows():
                f.write(f"{row['Model/Analysis']:<30} {row['Method']:<20} {row['Interpretation']}\n")
            f.write("\n\n")
            
            # Table 4
            f.write("TABLE 4. Temperature-Health Association Results\n")
            f.write("-" * 50 + "\n")
            for _, row in table4.iterrows():
                f.write(f"{row['Pathway']:<15} {row['Analysis']:<25} Effect={row['Effect Size']:<10} p={row['P-value']}\n")
        
        print("   • Formatted tables created for publication")


def main():
    """Main execution function"""
    
    print("📊 Starting Publication Tables Generation")
    print("=" * 50)
    
    # Configuration
    config = {
        'gcro_data': '/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv',
        'dlnm_results': '/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/results/dlnm_comprehensive_report.json',
        'enhanced_results': '/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/results/enhanced_rigorous_analysis_report.json',
        'output_dir': '/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/results/'
    }
    
    try:
        # Initialize generator
        generator = PublicationTablesGenerator(
            config['gcro_data'],
            config['dlnm_results'],
            config['enhanced_results']
        )
        
        # Generate all tables
        tables = generator.save_all_tables(config['output_dir'])
        
        print("\n🎉 Publication Tables Generation Complete!")
        print("=" * 50)
        print("📋 Tables Generated:")
        print(f"   • Table 1: Population Characteristics ({len(tables['table1'])} rows)")
        print(f"   • Table 2: Climate Exposure Statistics ({len(tables['table2'])} rows)")
        print(f"   • Table 3: DLNM Model Results ({len(tables['table3'])} rows)")
        print(f"   • Table 4: Temperature-Health Associations ({len(tables['table4'])} rows)")
        
        print(f"\n📁 Files saved to: {config['output_dir']}")
        print("   • CSV files for data analysis")
        print("   • Formatted text file for manuscript")
        
        return generator, tables
        
    except Exception as e:
        print(f"\n❌ Table generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    generator, tables = main()