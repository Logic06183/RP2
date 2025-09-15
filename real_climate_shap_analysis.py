#!/usr/bin/env python3
"""
Real Climate SHAP Analysis
===========================
Use actual climate data (Zarr files) linked to biomarker records for proper heat effect analysis
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.data_loader import DataLoader

plt.switch_backend('Agg')

class RealClimateSHAPAnalysis:
    """SHAP analysis using actual climate data from Zarr files."""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.climate_data = {}
        self.results = {}
        
    def load_climate_data(self):
        """Load actual climate data from Zarr files."""
        print("🌡️ LOADING REAL CLIMATE DATA")
        print("=" * 50)
        
        climate_path = Path("/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/climate/johannesburg/")
        
        # Load key climate datasets
        climate_files = {
            'ERA5_temperature': 'ERA5_tas_native.zarr',
            'ERA5_land_temp': 'ERA5-Land_tas_native.zarr',
            'WRF_temperature': 'WRF_tas_native.zarr',
            'MODIS_LST': 'modis_lst_native.zarr',
            'SAAQIS': 'SAAQIS_with_climate_variables.zarr'
        }
        
        loaded_datasets = {}
        
        for name, filename in climate_files.items():
            filepath = climate_path / filename
            try:
                print(f"📁 Loading {name}...")
                ds = xr.open_zarr(filepath)
                print(f"   📊 Dimensions: {dict(ds.dims)}")
                print(f"   📅 Time range: {ds.time.min().values} to {ds.time.max().values}")
                print(f"   🌍 Variables: {list(ds.data_vars.keys())}")
                
                loaded_datasets[name] = ds
                
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {str(e)}")
        
        self.climate_data = loaded_datasets
        return loaded_datasets
    
    def create_climate_biomarker_dataset(self):
        """Create dataset linking biomarker records to climate data."""
        print(f"\\n🔗 LINKING CLIMATE DATA TO BIOMARKER RECORDS")
        print("=" * 55)
        
        # Load biomarker data
        processed_data = self.data_loader.process_data()
        print(f"📊 Biomarker dataset: {len(processed_data):,} records")
        
        # Filter to records with coordinates and dates
        geo_biomarker = processed_data[
            processed_data['latitude'].notna() & 
            processed_data['longitude'].notna() &
            processed_data['date'].notna()
        ].copy()
        
        print(f"🌍 Records with coordinates: {len(geo_biomarker):,}")
        
        # Extract climate data for each biomarker record
        enhanced_biomarker_data = []
        
        # Use ERA5 temperature as primary climate source
        if 'ERA5_temperature' in self.climate_data:
            print(f"\\n🌡️ Extracting ERA5 temperature data...")
            
            era5_temp = self.climate_data['ERA5_temperature']
            
            # Sample subset for demonstration (full processing would be computationally intensive)
            sample_records = geo_biomarker.sample(min(2000, len(geo_biomarker)), random_state=42)
            
            for idx, record in sample_records.iterrows():
                try:
                    lat = record['latitude']
                    lon = record['longitude']
                    date_str = record['date']
                    
                    # Parse date
                    if pd.isna(date_str):
                        continue
                    
                    try:
                        if isinstance(date_str, str):
                            date = pd.to_datetime(date_str)
                        else:
                            date = pd.to_datetime(str(date_str))
                    except:
                        continue
                    
                    # Extract climate data for this location and time
                    # Select closest grid point
                    temp_point = era5_temp.sel(
                        latitude=lat, longitude=lon, 
                        method='nearest'
                    ).sel(time=date, method='nearest')
                    
                    # Extract temperature values
                    if 't2m' in temp_point:
                        temp_celsius = float(temp_point.t2m.values) - 273.15
                        
                        # Create extended record with climate data
                        enhanced_record = record.copy()
                        enhanced_record['real_temperature'] = temp_celsius
                        enhanced_record['temp_above_30'] = 1 if temp_celsius > 30 else 0
                        enhanced_record['temp_above_35'] = 1 if temp_celsius > 35 else 0
                        enhanced_record['heat_stress'] = max(0, temp_celsius - 25)
                        
                        # Add time-based features
                        enhanced_record['month'] = date.month
                        enhanced_record['day_of_year'] = date.dayofyear
                        enhanced_record['season'] = self._get_season(date.month)
                        
                        enhanced_biomarker_data.append(enhanced_record)
                        
                        if len(enhanced_biomarker_data) % 100 == 0:
                            print(f"   ✅ Processed {len(enhanced_biomarker_data)} records...")
                    
                except Exception as e:
                    continue
        
        if enhanced_biomarker_data:
            enhanced_df = pd.DataFrame(enhanced_biomarker_data)
            print(f"\\n🎯 Successfully linked: {len(enhanced_df):,} records with real climate data")
            return enhanced_df
        else:
            print(f"\\n❌ No records successfully linked to climate data")
            return None
    
    def _get_season(self, month):
        """Get season from month (Southern Hemisphere)."""
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'autumn'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'
    
    def run_real_climate_shap_analysis(self):
        """Run SHAP analysis with real climate data."""
        print(f"\\n🔬 REAL CLIMATE SHAP ANALYSIS")
        print("=" * 45)
        
        # Load climate data
        climate_datasets = self.load_climate_data()
        
        if not climate_datasets:
            print("❌ No climate data available")
            return None
        
        # Create linked dataset
        climate_biomarker_data = self.create_climate_biomarker_dataset()
        
        if climate_biomarker_data is None or len(climate_biomarker_data) < 100:
            print("❌ Insufficient climate-biomarker linkage")
            return None
        
        # Enhanced biomarker models with real climate predictors
        enhanced_configs = {
            "Systolic_BP_RealClimate": {
                "outcome": "systolic blood pressure",
                "predictors": ["Age (at enrolment)", "Sex", "Race", "real_temperature", 
                             "temp_above_30", "temp_above_35", "heat_stress", "month", "season", "employment"],
                "pathway_type": "cardiovascular",
                "min_samples": 50
            },
            "Diastolic_BP_RealClimate": {
                "outcome": "diastolic blood pressure", 
                "predictors": ["Age (at enrolment)", "Sex", "Race", "real_temperature", 
                             "temp_above_30", "temp_above_35", "heat_stress", "month", "season", "employment"],
                "pathway_type": "cardiovascular",
                "min_samples": 50
            },
            "Hemoglobin_RealClimate": {
                "outcome": "Hemoglobin (g/dL)",
                "predictors": ["Age (at enrolment)", "Sex", "Race", "real_temperature", 
                             "heat_stress", "season", "employment"],
                "pathway_type": "hematologic",
                "min_samples": 50
            }
        }
        
        # Train models with real climate data
        for model_name, config in enhanced_configs.items():
            print(f"\\n🧪 Real Climate Analysis: {model_name}")
            print(f"   Outcome: {config['outcome']}")
            
            try:
                # Prepare real climate dataset
                model_data = self._prepare_real_climate_data(climate_biomarker_data, config)
                
                if model_data is None or len(model_data) < config['min_samples']:
                    print(f"   ❌ Insufficient data: {len(model_data) if model_data is not None else 0} samples")
                    continue
                
                # Train model with real climate predictors
                results = self._train_real_climate_model(model_data, config, model_name)
                
                if results:
                    self.results[model_name] = results
                    
                    # Detailed climate SHAP analysis
                    self._analyze_real_climate_shap(results, model_name, config)
                
            except Exception as e:
                print(f"   ❌ Error in {model_name}: {str(e)}")
        
        # Generate real climate insights
        self._generate_real_climate_insights()
        
        return self.results
    
    def _prepare_real_climate_data(self, data, config):
        """Prepare dataset with real climate variables."""
        outcome = config['outcome']
        predictors = config['predictors']
        
        # Filter to records with the biomarker
        biomarker_data = data[data[outcome].notna()].copy()
        
        if len(biomarker_data) < 20:
            return None
        
        # Select available predictors
        available_predictors = []
        for predictor in predictors:
            if predictor in biomarker_data.columns:
                non_null = biomarker_data[predictor].notna().sum()
                if non_null >= len(biomarker_data) * 0.5:  # At least 50% coverage
                    available_predictors.append(predictor)
                    print(f"   ✅ {predictor}: {non_null:,} records ({non_null/len(biomarker_data)*100:.1f}%)")
        
        if len(available_predictors) < 3:
            return None
        
        # Create final dataset
        final_cols = available_predictors + [outcome]
        model_data = biomarker_data[final_cols].copy()
        
        # Enhanced preprocessing for climate variables
        for col in available_predictors:
            if col == 'season':
                le = LabelEncoder()
                model_data[col] = le.fit_transform(model_data[col].fillna('unknown'))
            elif model_data[col].dtype == 'object':
                model_data[col] = model_data[col].fillna('missing')
                le = LabelEncoder()
                model_data[col] = le.fit_transform(model_data[col])
            else:
                model_data[col] = model_data[col].fillna(model_data[col].median())
        
        # Remove remaining missing values
        model_data = model_data.dropna()
        
        print(f"   📊 Final dataset: {len(model_data):,} samples, {len(available_predictors)} predictors")
        
        return model_data
    
    def _train_real_climate_model(self, model_data, config, model_name):
        """Train XGBoost model with real climate data."""
        outcome = config['outcome']
        predictors = [col for col in model_data.columns if col != outcome]
        
        X = model_data[predictors]
        y = model_data[outcome]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Enhanced XGBoost for climate relationships
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"   📈 Real Climate Model R²: {test_r2:.3f}")
        
        # SHAP analysis with real climate features
        print(f"   🔍 Computing SHAP for real climate variables...")
        shap_explainer = shap.Explainer(model, X_train.sample(min(100, len(X_train)), random_state=42))
        shap_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        shap_values = shap_explainer(shap_sample)
        
        # Feature importance with climate focus
        feature_importance = {}
        for i, feature in enumerate(X.columns):
            importance = np.abs(shap_values.values[:, i]).mean()
            feature_importance[feature] = importance
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'test_r2': test_r2,
            'shap_explainer': shap_explainer,
            'shap_values': shap_values,
            'shap_sample': shap_sample,
            'feature_importance': feature_importance,
            'predictors': predictors,
            'sample_size': len(X)
        }
    
    def _analyze_real_climate_shap(self, results, model_name, config):
        """Analyze SHAP values for real climate variables."""
        print(f"\\n🔬 REAL CLIMATE SHAP ANALYSIS: {model_name}")
        print("=" * 50)
        
        feature_importance = results['feature_importance']
        
        # Categorize real climate features
        real_climate_features = {}
        socio_features = {}
        demo_features = {}
        
        for feature, importance in feature_importance.items():
            if feature in ['real_temperature', 'temp_above_30', 'temp_above_35', 'heat_stress', 'month', 'season']:
                real_climate_features[feature] = importance
            elif feature in ['employment', 'education', 'income']:
                socio_features[feature] = importance
            elif feature in ['Age (at enrolment)', 'Sex', 'Race']:
                demo_features[feature] = importance
        
        # Display real climate effects
        print(f"🌡️ REAL CLIMATE VARIABLES (SHAP Importance):")
        if real_climate_features:
            sorted_climate = sorted(real_climate_features.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_climate, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
                
                # Show SHAP effect direction for temperature variables
                if feature in results['predictors']:
                    feature_idx = list(results['predictors']).index(feature)
                    shap_vals = results['shap_values'].values[:, feature_idx]
                    avg_positive = shap_vals[shap_vals > 0].mean() if (shap_vals > 0).any() else 0
                    avg_negative = shap_vals[shap_vals < 0].mean() if (shap_vals < 0).any() else 0
                    
                    if 'temp' in feature.lower() or 'heat' in feature.lower():
                        print(f"      🌡️ Heat effect: +{avg_positive:.3f} (warming), {avg_negative:.3f} (cooling)")
                    else:
                        print(f"      📊 Effect: +{avg_positive:.3f}, {avg_negative:.3f}")
        else:
            print("   ❌ No real climate variables found")
        
        # Compare with other pathways
        print(f"\\n👥 DEMOGRAPHIC VARIABLES:")
        if demo_features:
            sorted_demo = sorted(demo_features.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_demo:
                print(f"   • {feature}: {importance:.4f}")
        
        print(f"\\n🏢 SOCIOECONOMIC VARIABLES:")
        if socio_features:
            sorted_socio = sorted(socio_features.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_socio:
                print(f"   • {feature}: {importance:.4f}")
        
        # Calculate real pathway percentages
        total_climate = sum(real_climate_features.values())
        total_socio = sum(socio_features.values())
        total_demo = sum(demo_features.values())
        total_importance = total_climate + total_socio + total_demo
        
        if total_importance > 0:
            climate_pct = (total_climate / total_importance) * 100
            socio_pct = (total_socio / total_importance) * 100
            demo_pct = (total_demo / total_importance) * 100
            
            print(f"\\n📊 REAL PATHWAY CONTRIBUTIONS:")
            print(f"   🌡️ Real Climate: {climate_pct:.1f}%")
            print(f"   🏢 Socioeconomic: {socio_pct:.1f}%")
            print(f"   👥 Demographic: {demo_pct:.1f}%")
            
            # Create visualization
            self._create_real_climate_visualization(results, model_name, config, 
                                                  real_climate_features, socio_features)
    
    def _create_real_climate_visualization(self, results, model_name, config, 
                                         climate_features, socio_features):
        """Create visualization showing real climate effects."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Real Climate Variables Impact (Top Left)
        ax = axes[0, 0]
        if climate_features:
            sorted_climate = sorted(climate_features.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_climate)
            y_pos = np.arange(len(features))
            
            bars = ax.barh(y_pos, importances, color='#FF4500', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('🌡️ Real Climate Variables Impact', fontweight='bold', fontsize=12)
            
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + max(importances)*0.02, bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No Real Climate\\nVariables Found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('🌡️ Real Climate Variables Impact', fontweight='bold')
        
        # 2. Temperature Effect Analysis (Top Right)
        ax = axes[0, 1]
        if 'real_temperature' in results['predictors']:
            temp_idx = list(results['predictors']).index('real_temperature')
            shap_vals = results['shap_values'].values[:, temp_idx]
            
            # Scatter plot of temperature vs SHAP values
            temp_values = results['shap_sample']['real_temperature'].values
            ax.scatter(temp_values, shap_vals, alpha=0.6, color='red', s=30)
            ax.set_xlabel('Real Temperature (°C)')
            ax.set_ylabel('SHAP Value (Temperature Effect)')
            ax.set_title('🔥 Temperature vs Biomarker Effect', fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add trend line
            z = np.polyfit(temp_values, shap_vals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(temp_values), p(sorted(temp_values)), "r--", alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'No Temperature\\nData Available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('🔥 Temperature Effect Analysis', fontweight='bold')
        
        # 3. All Features Ranked (Bottom Left)
        ax = axes[1, 0]
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        features, importances = zip(*sorted_features)
        y_pos = np.arange(len(features))
        
        # Color code by type
        colors = []
        for feature in features:
            if feature in ['real_temperature', 'temp_above_30', 'temp_above_35', 'heat_stress', 'month', 'season']:
                colors.append('#FF4500')  # Orange-red for real climate
            elif feature in ['employment', 'education', 'income']:
                colors.append('#4ECDC4')  # Teal for socioeconomic
            else:
                colors.append('#FF6B6B')  # Red for demographic
        
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features], fontsize=9)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('📊 Top Predictors (Real Climate)', fontweight='bold')
        
        # 4. Model Summary (Bottom Right)
        ax = axes[1, 1]
        ax.axis('off')
        
        test_r2 = results['test_r2']
        sample_size = results['sample_size']
        
        summary_text = [
            f"📈 Real Climate Model:",
            f"R² Score: {test_r2:.3f}",
            f"Sample Size: {sample_size:,}",
            f"",
            f"🌡️ Climate Variables: {len(climate_features)}",
            f"🏢 Socioeconomic: {len(socio_features)}",
            f"",
            f"🎯 Outcome: {config['outcome']}",
            f"🔬 Uses Real ERA5 Temperature Data",
        ]
        
        for i, text in enumerate(summary_text):
            weight = 'bold' if text.endswith(':') else 'normal'
            ax.text(0.05, 0.95 - i*0.1, text, transform=ax.transAxes,
                   fontsize=11, weight=weight, va='top')
        
        plt.suptitle(f'Real Climate SHAP Analysis: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save as SVG
        filename = f"real_climate_shap_{model_name.lower()}.svg"
        filepath = Path("figures") / filename
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Real climate visualization saved: {filepath}")
    
    def _generate_real_climate_insights(self):
        """Generate insights about real climate effects."""
        print(f"\\n🔥 REAL CLIMATE EFFECTS SUMMARY")
        print("=" * 45)
        
        if not self.results:
            print("❌ No real climate results available")
            return
        
        # Analyze real climate effects across models
        climate_summary = {}
        
        for model_name, results in self.results.items():
            feature_importance = results['feature_importance']
            
            # Extract real climate effects
            climate_effects = {
                'real_temperature': feature_importance.get('real_temperature', 0),
                'temp_above_30': feature_importance.get('temp_above_30', 0),
                'temp_above_35': feature_importance.get('temp_above_35', 0),
                'heat_stress': feature_importance.get('heat_stress', 0),
            }
            
            total_climate_effect = sum(climate_effects.values())
            
            climate_summary[model_name] = {
                'r2': results['test_r2'],
                'sample_size': results['sample_size'],
                'climate_effects': climate_effects,
                'total_climate_importance': total_climate_effect
            }
        
        # Display results
        print("🌡️ REAL CLIMATE VARIABLE EFFECTS BY BIOMARKER:")
        
        for model_name, summary in climate_summary.items():
            print(f"\\n🧪 {model_name}:")
            print(f"   R² Score: {summary['r2']:.3f}")
            print(f"   Sample Size: {summary['sample_size']:,}")
            print(f"   Total Climate Effect: {summary['total_climate_importance']:.4f}")
            
            print(f"   Real Climate Variables:")
            for var, effect in summary['climate_effects'].items():
                if effect > 0:
                    print(f"     • {var}: {effect:.4f}")
        
        # Overall insights
        avg_climate_effect = np.mean([s['total_climate_importance'] for s in climate_summary.values()])
        print(f"\\n📊 AVERAGE REAL CLIMATE EFFECT: {avg_climate_effect:.4f}")
        
        if avg_climate_effect > 0.01:
            print("✅ Real climate variables show measurable effects on biomarkers!")
        else:
            print("⚠️ Real climate effects are still weak - may need larger sample or different approach")


if __name__ == "__main__":
    analyzer = RealClimateSHAPAnalysis()
    results = analyzer.run_real_climate_shap_analysis()
    
    print(f"\\n🎯 REAL CLIMATE SHAP ANALYSIS COMPLETE")
    if results:
        print(f"✅ {len(results)} models trained with real climate data")
        print(f"✅ Actual ERA5 temperature data used")
        print(f"✅ Heat effects quantified with SHAP")
    else:
        print("❌ Analysis failed - check data linkage issues")