#!/usr/bin/env python3
"""
Detailed SHAP Pathway Analysis
==============================
Deep dive into SHAP values for climate and socioeconomic variables
focusing on heat effects across all biomarker pathways
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.data_loader import DataLoader

plt.switch_backend('Agg')

class DetailedSHAPPathwayAnalysis:
    """Detailed analysis of SHAP values by pathway with focus on climate/socioeconomic effects."""
    
    def __init__(self):
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.results = {}
        
    def run_detailed_pathway_analysis(self):
        """Run detailed SHAP analysis focusing on climate and socioeconomic pathways."""
        print("🔍 DETAILED SHAP PATHWAY ANALYSIS")
        print("="*70)
        print("Deep dive into climate & socioeconomic effects on biomarkers")
        print()
        
        # Load and process data
        processed_data = self.data_loader.process_data()
        print(f"📊 Dataset: {len(processed_data):,} records, {len(processed_data.columns)} features")
        
        # First, let's examine what climate variables are actually available
        climate_vars = [col for col in processed_data.columns if any(term in col.lower() 
                       for term in ['temp', 'heat', 'cool', 'era5', 'climate'])]
        print(f"\n🌡️  Available Climate Variables ({len(climate_vars)}):")
        for i, var in enumerate(climate_vars[:15], 1):
            coverage = processed_data[var].notna().sum()
            print(f"   {i:2d}. {var}: {coverage:,} records ({coverage/len(processed_data)*100:.1f}%)")
        
        # Examine socioeconomic variables
        socio_vars = [col for col in processed_data.columns if any(term in col.lower() 
                     for term in ['income', 'employ', 'education', 'municipality', 'ward'])]
        print(f"\n🏢 Available Socioeconomic Variables ({len(socio_vars)}):")
        for i, var in enumerate(socio_vars[:15], 1):
            coverage = processed_data[var].notna().sum()
            print(f"   {i:2d}. {var}: {coverage:,} records ({coverage/len(processed_data)*100:.1f}%)")
        
        # Enhanced biomarker configurations with expanded climate predictors
        enhanced_configs = {
            "Systolic_BP_Enhanced": {
                "outcome": "systolic blood pressure",
                "predictors": ["Age (at enrolment)", "Sex", "Race"] + climate_vars[:8] + socio_vars[:5],
                "pathway_type": "cardiovascular",
                "min_samples": 1000
            },
            "Diastolic_BP_Enhanced": {
                "outcome": "diastolic blood pressure", 
                "predictors": ["Age (at enrolment)", "Sex", "Race"] + climate_vars[:8] + socio_vars[:5],
                "pathway_type": "cardiovascular",
                "min_samples": 1000
            },
            "Hemoglobin_Enhanced": {
                "outcome": "Hemoglobin (g/dL)",
                "predictors": ["Age (at enrolment)", "Sex", "Race"] + climate_vars[:6] + socio_vars[:4],
                "pathway_type": "hematologic",
                "min_samples": 200
            }
        }
        
        # Train enhanced models with more climate variables
        for model_name, config in enhanced_configs.items():
            print(f"\n🧪 Enhanced Analysis: {model_name}")
            print(f"   Outcome: {config['outcome']}")
            print(f"   Pathway: {config['pathway_type']}")
            
            try:
                # Prepare enhanced dataset
                model_data = self._prepare_enhanced_biomarker_data(processed_data, config)
                
                if model_data is None or len(model_data) < config['min_samples']:
                    print(f"   ❌ Insufficient data: {len(model_data) if model_data is not None else 0} samples")
                    continue
                
                # Train model with enhanced predictors
                results = self._train_enhanced_model(model_data, config, model_name)
                
                if results:
                    self.results[model_name] = results
                    
                    # Detailed SHAP analysis by pathway
                    self._analyze_detailed_shap_pathways(results, model_name, config)
                
            except Exception as e:
                print(f"   ❌ Error in {model_name}: {str(e)}")
        
        # Generate heat-specific insights
        self._generate_heat_specific_insights()
        
        return self.results
    
    def _prepare_enhanced_biomarker_data(self, data, config):
        """Prepare enhanced biomarker dataset with better predictor selection."""
        outcome = config['outcome']
        candidate_predictors = config['predictors']
        
        # Filter to records with the biomarker
        biomarker_data = data[data[outcome].notna()].copy()
        
        if len(biomarker_data) < 50:
            return None
        
        # Select predictors with good coverage
        selected_predictors = []
        for predictor in candidate_predictors:
            if predictor in biomarker_data.columns:
                non_null = biomarker_data[predictor].notna().sum()
                coverage = non_null / len(biomarker_data)
                
                # More lenient coverage threshold for climate variables
                min_coverage = 0.05 if any(term in predictor.lower() 
                                         for term in ['temp', 'heat', 'era5']) else 0.1
                
                if coverage >= min_coverage and non_null >= 100:
                    selected_predictors.append(predictor)
                    print(f"   ✅ Selected {predictor}: {coverage*100:.1f}% coverage ({non_null:,} records)")
        
        if len(selected_predictors) < 5:
            print(f"   ⚠️  Only {len(selected_predictors)} predictors available")
            return None
        
        # Create final dataset
        final_cols = selected_predictors + [outcome]
        model_data = biomarker_data[final_cols].copy()
        
        # Enhanced preprocessing
        for col in selected_predictors:
            if model_data[col].dtype == 'object':
                model_data[col] = model_data[col].fillna('missing')
                le = LabelEncoder()
                try:
                    model_data[col] = le.fit_transform(model_data[col].astype(str))
                except:
                    model_data[col] = 0
            else:
                # For numeric variables, use median imputation
                median_val = model_data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                model_data[col] = model_data[col].fillna(median_val)
        
        # Remove any remaining missing values
        initial_size = len(model_data)
        model_data = model_data.dropna()
        
        print(f"   📊 Data prepared: {len(model_data):,} samples ({initial_size-len(model_data)} dropped)")
        print(f"   🔧 Final predictors ({len(selected_predictors)}): {selected_predictors[:10]}{'...' if len(selected_predictors) > 10 else ''}")
        
        return model_data
    
    def _train_enhanced_model(self, model_data, config, model_name):
        """Train enhanced XGBoost model with detailed SHAP analysis."""
        outcome = config['outcome']
        predictors = [col for col in model_data.columns if col != outcome]
        
        X = model_data[predictors]
        y = model_data[outcome]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Enhanced XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        
        print(f"   📈 Enhanced Model R²: {test_r2:.3f}")
        
        # Comprehensive SHAP analysis
        print(f"   🔍 Computing detailed SHAP values...")
        shap_explainer = shap.Explainer(model, X_train.sample(min(200, len(X_train)), random_state=42))
        shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = shap_explainer(shap_sample)
        
        # Detailed feature importance
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
    
    def _analyze_detailed_shap_pathways(self, results, model_name, config):
        """Analyze SHAP values by pathway with top 10 for each category."""
        print(f"\n🔬 DETAILED SHAP PATHWAY ANALYSIS: {model_name}")
        print("=" * 60)
        
        feature_importance = results['feature_importance']
        
        # Categorize features
        climate_features = {}
        socio_features = {}
        demo_features = {}
        
        for feature, importance in feature_importance.items():
            if any(term in feature.lower() for term in ['temp', 'heat', 'cool', 'era5', 'climate']):
                climate_features[feature] = importance
            elif any(term in feature.lower() for term in ['income', 'employ', 'education', 'municipality', 'ward']):
                socio_features[feature] = importance
            elif any(term in feature.lower() for term in ['age', 'sex', 'race']):
                demo_features[feature] = importance
        
        # Display top 10 for each pathway
        print(f"🌡️  TOP CLIMATE VARIABLES (SHAP Importance):")
        if climate_features:
            sorted_climate = sorted(climate_features.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_climate, 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
                
                # Show actual SHAP value distribution for top climate features
                if i <= 3:
                    feature_idx = list(results['predictors']).index(feature)
                    shap_vals = results['shap_values'].values[:, feature_idx]
                    avg_positive = shap_vals[shap_vals > 0].mean() if (shap_vals > 0).any() else 0
                    avg_negative = shap_vals[shap_vals < 0].mean() if (shap_vals < 0).any() else 0
                    print(f"       📊 SHAP effect: +{avg_positive:.3f} (warming), {avg_negative:.3f} (cooling)")
        else:
            print("   ❌ No climate variables found in top predictors")
        
        print(f"\n🏢 TOP SOCIOECONOMIC VARIABLES (SHAP Importance):")
        if socio_features:
            sorted_socio = sorted(socio_features.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_socio, 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
                
                # Show SHAP effects for top socioeconomic features
                if i <= 3:
                    feature_idx = list(results['predictors']).index(feature)
                    shap_vals = results['shap_values'].values[:, feature_idx]
                    avg_positive = shap_vals[shap_vals > 0].mean() if (shap_vals > 0).any() else 0
                    avg_negative = shap_vals[shap_vals < 0].mean() if (shap_vals < 0).any() else 0
                    print(f"       📊 SHAP effect: +{avg_positive:.3f} (advantage), {avg_negative:.3f} (disadvantage)")
        else:
            print("   ❌ No socioeconomic variables found in top predictors")
        
        print(f"\n👥 TOP DEMOGRAPHIC VARIABLES (SHAP Importance):")
        if demo_features:
            sorted_demo = sorted(demo_features.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_demo, 1):
                print(f"   {i:2d}. {feature}: {importance:.4f}")
        else:
            print("   ❌ No demographic variables found in top predictors")
        
        # Calculate pathway percentages
        total_climate = sum(climate_features.values())
        total_socio = sum(socio_features.values())
        total_demo = sum(demo_features.values())
        total_importance = total_climate + total_socio + total_demo
        
        if total_importance > 0:
            climate_pct = (total_climate / total_importance) * 100
            socio_pct = (total_socio / total_importance) * 100
            demo_pct = (total_demo / total_importance) * 100
            
            print(f"\n📊 PATHWAY CONTRIBUTIONS:")
            print(f"   🌡️  Climate: {climate_pct:.1f}% ({len(climate_features)} variables)")
            print(f"   🏢 Socioeconomic: {socio_pct:.1f}% ({len(socio_features)} variables)")
            print(f"   👥 Demographic: {demo_pct:.1f}% ({len(demo_features)} variables)")
        
        # Create detailed pathway visualization
        self._create_detailed_pathway_plot(results, model_name, config, climate_features, socio_features)
    
    def _create_detailed_pathway_plot(self, results, model_name, config, climate_features, socio_features):
        """Create detailed pathway visualization focusing on climate and socioeconomic effects."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Climate Variables SHAP Plot (Top Left)
        ax = axes[0, 0]
        if climate_features:
            sorted_climate = sorted(climate_features.items(), key=lambda x: x[1], reverse=True)[:8]
            features, importances = zip(*sorted_climate)
            y_pos = np.arange(len(features))
            
            bars = ax.barh(y_pos, importances, color='#45B7D1', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('🌡️ Climate Variables Impact', fontweight='bold', fontsize=12)
            
            # Add value labels
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + max(importances)*0.02, bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}', ha='left', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Climate Variables\nFound in Top Predictors', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('🌡️ Climate Variables Impact', fontweight='bold')
        
        # 2. Socioeconomic Variables SHAP Plot (Top Right)
        ax = axes[0, 1]
        if socio_features:
            sorted_socio = sorted(socio_features.items(), key=lambda x: x[1], reverse=True)[:8]
            features, importances = zip(*sorted_socio)
            y_pos = np.arange(len(features))
            
            bars = ax.barh(y_pos, importances, color='#4ECDC4', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.set_title('🏢 Socioeconomic Variables Impact', fontweight='bold', fontsize=12)
            
            # Add value labels
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + max(importances)*0.02, bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f}', ha='left', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Socioeconomic Variables\nFound in Top Predictors', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('🏢 Socioeconomic Variables Impact', fontweight='bold')
        
        # 3. Overall Feature Importance (Bottom Left)
        ax = axes[1, 0]
        feature_importance = results['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
        
        features, importances = zip(*sorted_features)
        y_pos = np.arange(len(features))
        
        # Color code by pathway
        colors = []
        for feature in features:
            if any(term in feature.lower() for term in ['temp', 'heat', 'cool', 'era5']):
                colors.append('#45B7D1')  # Blue for climate
            elif any(term in feature.lower() for term in ['income', 'employ', 'education', 'municipality']):
                colors.append('#4ECDC4')  # Teal for socioeconomic
            else:
                colors.append('#FF6B6B')  # Red for demographic
        
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in features], fontsize=8)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('📊 Top 12 Overall Predictors', fontweight='bold', fontsize=12)
        
        # 4. Model Performance Summary (Bottom Right)
        ax = axes[1, 1]
        ax.axis('off')
        
        # Performance metrics
        test_r2 = results['test_r2']
        sample_size = results['sample_size']
        
        summary_text = [
            f"📈 Model Performance:",
            f"R² Score: {test_r2:.3f}",
            f"Sample Size: {sample_size:,}",
            f"Predictors: {len(results['predictors'])}",
            "",
            f"🔍 Climate Variables Found: {len(climate_features)}",
            f"🏢 Socioeconomic Variables: {len(socio_features)}",
            "",
            f"🎯 Outcome: {config['outcome']}",
            f"🛤️ Pathway: {config['pathway_type'].title()}",
        ]
        
        for i, text in enumerate(summary_text):
            weight = 'bold' if text.endswith(':') else 'normal'
            ax.text(0.05, 0.95 - i*0.08, text, transform=ax.transAxes,
                   fontsize=11, weight=weight, va='top')
        
        plt.suptitle(f'Detailed Pathway Analysis: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save as SVG
        filename = f"detailed_pathway_{model_name.lower()}.svg"
        filepath = Path("figures") / filename
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Detailed pathway plot saved: {filepath}")
    
    def _generate_heat_specific_insights(self):
        """Generate specific insights about heat effects on biomarkers."""
        print(f"\n🔥 HEAT-SPECIFIC INSIGHTS ACROSS PATHWAYS")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results available for heat analysis")
            return
        
        # Analyze heat effects across all models
        heat_effects = {}
        
        for model_name, results in self.results.items():
            feature_importance = results['feature_importance']
            
            # Find heat-related variables
            heat_vars = {k: v for k, v in feature_importance.items() 
                        if any(term in k.lower() for term in ['temp', 'heat'])}
            
            if heat_vars:
                heat_effects[model_name] = {
                    'biomarker': model_name,
                    'r2': results['test_r2'],
                    'heat_variables': heat_vars,
                    'total_heat_importance': sum(heat_vars.values()),
                    'sample_size': results['sample_size']
                }
        
        if heat_effects:
            print("🌡️ HEAT VARIABLE EFFECTS BY BIOMARKER:")
            
            # Sort by total heat importance
            sorted_heat = sorted(heat_effects.items(), key=lambda x: x[1]['total_heat_importance'], reverse=True)
            
            for model_name, effects in sorted_heat:
                print(f"\n🧪 {model_name}:")
                print(f"   R² Score: {effects['r2']:.3f}")
                print(f"   Sample Size: {effects['sample_size']:,}")
                print(f"   Total Heat Importance: {effects['total_heat_importance']:.4f}")
                print(f"   Heat Variables Found:")
                
                for heat_var, importance in sorted(effects['heat_variables'].items(), key=lambda x: x[1], reverse=True):
                    print(f"     • {heat_var}: {importance:.4f}")
        
        else:
            print("❌ No significant heat effects found in any biomarker models")
            print("\n🤔 POTENTIAL REASONS:")
            print("   • Climate variables may have poor data coverage")
            print("   • Heat effects might be too small relative to demographic factors")
            print("   • Temporal misalignment between climate and health data")
            print("   • Non-linear relationships not captured by current features")


if __name__ == "__main__":
    analyzer = DetailedSHAPPathwayAnalysis()
    results = analyzer.run_detailed_pathway_analysis()
    
    print(f"\n🎯 DETAILED SHAP PATHWAY ANALYSIS COMPLETE")
    print(f"✅ Enhanced models with expanded climate predictors")
    print(f"✅ Top 10 SHAP values analyzed for each pathway")
    print(f"✅ Heat-specific effects investigated")
    print(f"✅ Detailed pathway visualizations created")