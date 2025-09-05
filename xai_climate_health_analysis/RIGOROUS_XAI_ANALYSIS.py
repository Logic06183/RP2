#!/usr/bin/env python3
"""
Rigorous XAI Climate-Health Analysis
Author: Craig Parker
High-performance analysis for top-tier journal publication
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and XAI
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
from scipy import stats

class RigorousXAIAnalysis:
    def __init__(self):
        self.data_dir = Path("xai_climate_health_analysis/data")
        self.results_dir = Path("rigorous_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and integrate all datasets efficiently"""
        # Load health data
        health_files = list(self.data_dir.glob("health/*.csv"))
        health_data = []
        
        for file in health_files:
            try:
                df = pd.read_csv(file)
                # Standardize date column
                if 'visit_date' in df.columns:
                    df['date'] = df['visit_date']
                elif 'date' not in df.columns:
                    continue
                    
                if len(df) > 100:  # Include more cohorts
                    health_data.append(df)
            except Exception as e:
                continue
        
        if not health_data:
            raise ValueError("No health data loaded")
                
        health_df = pd.concat(health_data, ignore_index=True, sort=False)
        
        # Load real climate data
        import xarray as xr
        try:
            climate_ds = xr.open_zarr('/home/cparker/selected_data_all/data/RP2_subsets/JHB/ERA5_tas_native.zarr')
            climate_df = climate_ds.to_dataframe().reset_index()
            climate_df['date'] = pd.to_datetime(climate_df['time']).dt.date
            climate_daily = climate_df.groupby('date').agg({'tas': ['mean', 'max', 'min', 'std']}).reset_index()
            climate_daily.columns = ['date', 'temp_mean', 'temp_max', 'temp_min', 'temp_std']
        except Exception as e:
            print(f"Climate data loading failed: {e}")
            return None
        
        # Merge datasets with proper date handling
        health_df['date'] = pd.to_datetime(health_df['date']).dt.date
        merged_df = pd.merge(health_df, climate_daily, on='date', how='inner')
        
        return merged_df
    
    def analyze_biomarker(self, data, biomarker):
        """Rigorous XAI analysis for single biomarker"""
        if biomarker not in data.columns:
            return None
            
        # Prepare features and target
        features = ['temp_mean', 'temp_max', 'temp_min']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) == 0:
            return None
            
        X = data[available_features].dropna()
        y = data[biomarker].loc[X.index].dropna()
        
        if len(y) < 100:
            return None
            
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[:500])  # Sample for efficiency
        
        # Calculate metrics
        r2 = r2_score(y, model.predict(X))
        
        return {
            'biomarker': biomarker,
            'r2_score': r2,
            'n_samples': len(y),
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else [],
            'feature_importance': dict(zip(available_features, model.feature_importances_)),
            'model_predictions': model.predict(X).tolist()
        }
    
    def create_visualizations(self, results):
        """Generate publication-quality SVG visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # Results summary plot
        biomarkers = [r['biomarker'] for r in results if r]
        r2_scores = [r['r2_score'] for r in results if r]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(biomarkers, r2_scores, color='steelblue', alpha=0.7)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_xlabel('Biomarker', fontsize=12)
        ax.set_title('Climate-Health XAI Analysis Results', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'xai_results_summary.svg', format='svg', dpi=300)
        plt.close()
        
        return str(self.results_dir / 'xai_results_summary.svg')
    
    def run_analysis(self):
        """Execute complete rigorous analysis"""
        print("Loading data...")
        data = self.load_data()
        print(f"Loaded {len(data)} records")
        
        # Identify biomarkers (actual column names from data)
        potential_biomarkers = ['FASTING GLUCOSE', 'FASTING TOTAL CHOLESTEROL', 'FASTING HDL', 
                               'FASTING LDL', 'systolic blood pressure', 'CREATININE', 'HEMOGLOBIN',
                               'CD4 Count', 'ALT', 'glucose', 'total_cholesterol', 'hdl_cholesterol']
        
        biomarkers = [b for b in potential_biomarkers if b in data.columns]
        print(f"Found biomarkers: {biomarkers}")
        
        results = []
        print("Running XAI analysis...")
        
        for biomarker in biomarkers:
            print(f"  Analyzing {biomarker}...")
            result = self.analyze_biomarker(data, biomarker)
            if result:
                results.append(result)
        
        # Generate visualizations
        print("Creating visualizations...")
        viz_file = self.create_visualizations(results)
        
        # Save results
        results_file = self.results_dir / 'xai_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis complete! Results saved to {results_file}")
        print(f"Visualization saved to {viz_file}")
        
        return results, viz_file

if __name__ == "__main__":
    analyzer = RigorousXAIAnalysis()
    results, viz_file = analyzer.run_analysis()