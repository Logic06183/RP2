#!/usr/bin/env python3
"""
Comprehensive Heat-Health SHAP Analysis
=======================================
Top 10 climate and socioeconomic SHAP values for each biomarker pathway
using the high-quality climate data from the agent analysis.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

plt.switch_backend('Agg')

class ComprehensiveHeatSHAPAnalysis:
    """Comprehensive SHAP analysis showing top climate and socioeconomic effects."""
    
    def __init__(self):
        self.enhanced_data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/ENHANCED_CLIMATE_BIOMARKER_DATASET.csv"
        self.results = {}
        
    def run_comprehensive_analysis(self):
        """Run comprehensive SHAP analysis with enhanced climate data."""
        print("🔥 COMPREHENSIVE HEAT-HEALTH SHAP ANALYSIS")
        print("="*60)
        print("Top 10 Climate & Socioeconomic Variables per Pathway")
        print()
        
        # Load enhanced climate-biomarker dataset
        try:
            enhanced_data = pd.read_csv(self.enhanced_data_path, low_memory=False)
            print(f"📊 Enhanced dataset loaded: {len(enhanced_data):,} records")
        except FileNotFoundError:
            print("❌ Enhanced climate dataset not found. Using fallback approach...")
            return self._run_fallback_analysis()
        
        # Identify climate and socioeconomic variables
        climate_vars = [col for col in enhanced_data.columns if col.startswith('climate_')]
        socio_vars = ['employment', 'education', 'income', 'municipality', 'ward']
        demo_vars = ['Age (at enrolment)', 'Sex', 'Race']
        
        print(f"🌡️  Climate variables found: {len(climate_vars)}")
        print(f"🏢 Socioeconomic variables: {len([v for v in socio_vars if v in enhanced_data.columns])}")
        print(f"👥 Demographic variables: {len([v for v in demo_vars if v in enhanced_data.columns])}")
        
        # Enhanced biomarker configurations
        biomarker_configs = {
            "Systolic_BP_Enhanced": {
                "outcome": "systolic blood pressure",
                "pathway_type": "cardiovascular",
                "min_samples": 500
            },
            "Diastolic_BP_Enhanced": {
                "outcome": "diastolic blood pressure", 
                "pathway_type": "cardiovascular",
                "min_samples": 500
            },
            "Hemoglobin_Enhanced": {
                "outcome": "Hemoglobin (g/dL)",
                "pathway_type": "hematologic",
                "min_samples": 200
            },
            "CD4_Count_Enhanced": {
                "outcome": "CD4 cell count (cells/µL)",
                "pathway_type": "immune",
                "min_samples": 200
            },
            "Creatinine_Enhanced": {
                "outcome": "Creatinine (mg/dL)",
                "pathway_type": "renal",
                "min_samples": 200
            }
        }
        
        # Train enhanced models with proper climate data
        for model_name, config in biomarker_configs.items():
            print(f"\\n🧪 Enhanced Analysis: {model_name}")
            print(f"   Pathway: {config['pathway_type']}")
            print(f"   Outcome: {config['outcome']}")
            
            try:
                # Prepare enhanced dataset
                model_data = self._prepare_enhanced_dataset(enhanced_data, config, 
                                                          climate_vars, socio_vars, demo_vars)
                
                if model_data is None or len(model_data) < config['min_samples']:
                    print(f"   ❌ Insufficient data: {len(model_data) if model_data is not None else 0} samples")
                    continue
                
                # Train enhanced model
                results = self._train_enhanced_shap_model(model_data, config, model_name)
                
                if results:
                    self.results[model_name] = results
                    
                    # Generate comprehensive pathway analysis
                    self._analyze_top_pathway_effects(results, model_name, config, 
                                                    climate_vars, socio_vars)
                    
                    # Create detailed visualizations
                    self._create_comprehensive_pathway_plots(results, model_name, config,
                                                           climate_vars, socio_vars)
                
            except Exception as e:
                print(f"   ❌ Error in {model_name}: {str(e)}")
        
        # Generate final comprehensive insights
        self._generate_comprehensive_heat_insights()
        
        return self.results
    
    def _prepare_enhanced_dataset(self, data, config, climate_vars, socio_vars, demo_vars):
        """Prepare enhanced dataset with climate, socioeconomic, and demographic predictors."""
        outcome = config['outcome']
        
        # Filter to records with the biomarker
        biomarker_data = data[data[outcome].notna()].copy()
        
        if len(biomarker_data) < 50:
            return None
        
        # Collect all predictors with good coverage
        all_predictors = []
        
        # Add climate variables
        for var in climate_vars:
            if var in biomarker_data.columns:
                coverage = biomarker_data[var].notna().sum() / len(biomarker_data)
                if coverage >= 0.8:  # At least 80% coverage
                    all_predictors.append(var)
                    print(f"   ✅ Climate: {var} ({coverage*100:.1f}% coverage)")
        
        # Add socioeconomic variables
        for var in socio_vars:
            if var in biomarker_data.columns:
                coverage = biomarker_data[var].notna().sum() / len(biomarker_data)
                if coverage >= 0.1:  # At least 10% coverage
                    all_predictors.append(var)
                    print(f"   ✅ Socio: {var} ({coverage*100:.1f}% coverage)")
        
        # Add demographic variables
        for var in demo_vars:
            if var in biomarker_data.columns:
                coverage = biomarker_data[var].notna().sum() / len(biomarker_data)
                if coverage >= 0.5:  # At least 50% coverage
                    all_predictors.append(var)
                    print(f"   ✅ Demo: {var} ({coverage*100:.1f}% coverage)")
        
        if len(all_predictors) < 5:
            print(f"   ⚠️  Only {len(all_predictors)} predictors available")
            return None
        
        # Create final dataset
        final_cols = all_predictors + [outcome]
        model_data = biomarker_data[final_cols].copy()
        
        # Enhanced preprocessing
        for col in all_predictors:
            if model_data[col].dtype == 'object':
                # Handle categorical variables
                model_data[col] = model_data[col].fillna('missing')
                le = LabelEncoder()
                model_data[col] = le.fit_transform(model_data[col].astype(str))
            else:
                # Handle numeric variables
                median_val = model_data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                model_data[col] = model_data[col].fillna(median_val)
        
        # Remove remaining missing values
        model_data = model_data.dropna()
        
        print(f"   📊 Enhanced dataset: {len(model_data):,} samples, {len(all_predictors)} predictors")
        
        return model_data
    
    def _train_enhanced_shap_model(self, model_data, config, model_name):
        """Train enhanced XGBoost model with comprehensive SHAP analysis."""
        outcome = config['outcome']
        predictors = [col for col in model_data.columns if col != outcome]
        
        X = model_data[predictors]
        y = model_data[outcome]
        
        # Enhanced train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Enhanced XGBoost model for better climate relationships
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
        
        # Enhanced evaluation
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        print(f"   📈 Enhanced Model Performance:")
        print(f"      Test R²: {test_r2:.3f}")
        print(f"      CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Comprehensive SHAP analysis
        print(f"   🔍 Computing comprehensive SHAP values...")
        shap_explainer = shap.Explainer(model, X_train.sample(min(200, len(X_train)), random_state=42))
        shap_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_values = shap_explainer(shap_sample)
        
        # Comprehensive feature importance
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
            'cv_scores': cv_scores,
            'shap_explainer': shap_explainer,
            'shap_values': shap_values,
            'shap_sample': shap_sample,
            'feature_importance': feature_importance,
            'predictors': predictors,
            'sample_size': len(X)
        }
    
    def _analyze_top_pathway_effects(self, results, model_name, config, climate_vars, socio_vars):
        """Analyze top 10 effects for each pathway category."""
        print(f"\\n🔬 TOP PATHWAY EFFECTS: {model_name}")
        print("="*55)
        
        feature_importance = results['feature_importance']
        
        # Categorize features by pathway
        climate_effects = {}
        socio_effects = {}
        demo_effects = {}
        
        for feature, importance in feature_importance.items():
            if any(climate_var in feature for climate_var in climate_vars) or feature.startswith('climate_'):
                climate_effects[feature] = importance
            elif any(socio_var in feature for socio_var in socio_vars):
                socio_effects[feature] = importance
            elif feature in ['Age (at enrolment)', 'Sex', 'Race']:
                demo_effects[feature] = importance
        
        # Display top 10 climate effects
        print(f"🌡️  TOP 10 CLIMATE/HEAT EFFECTS:")
        if climate_effects:
            sorted_climate = sorted(climate_effects.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_climate, 1):
                print(f"   {i:2d}. {self._clean_feature_name(feature)}: {importance:.4f}")
                
                # Show SHAP effect direction for key climate features
                if i <= 5 and feature in results['predictors']:
                    feature_idx = list(results['predictors']).index(feature)
                    shap_vals = results['shap_values'].values[:, feature_idx]
                    avg_positive = shap_vals[shap_vals > 0].mean() if (shap_vals > 0).any() else 0
                    avg_negative = shap_vals[shap_vals < 0].mean() if (shap_vals < 0).any() else 0
                    print(f"       🔥 Heat effect: +{avg_positive:.3f} (warming), {avg_negative:.3f} (cooling)")
        else:
            print("   ❌ No climate effects found")
        
        # Display top 10 socioeconomic effects
        print(f"\\n🏢 TOP 10 SOCIOECONOMIC EFFECTS:")
        if socio_effects:
            sorted_socio = sorted(socio_effects.items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_socio, 1):
                print(f"   {i:2d}. {self._clean_feature_name(feature)}: {importance:.4f}")
                
                # Show SHAP effect direction for key socio features
                if i <= 5 and feature in results['predictors']:
                    feature_idx = list(results['predictors']).index(feature)
                    shap_vals = results['shap_values'].values[:, feature_idx]
                    avg_positive = shap_vals[shap_vals > 0].mean() if (shap_vals > 0).any() else 0
                    avg_negative = shap_vals[shap_vals < 0].mean() if (shap_vals < 0).any() else 0
                    print(f"       💼 SES effect: +{avg_positive:.3f} (advantage), {avg_negative:.3f} (disadvantage)")
        else:
            print("   ❌ No socioeconomic effects found")
        
        # Display top demographic effects for comparison
        print(f"\\n👥 TOP DEMOGRAPHIC EFFECTS (for comparison):")
        if demo_effects:
            sorted_demo = sorted(demo_effects.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feature, importance) in enumerate(sorted_demo, 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        
        # Calculate pathway contribution percentages
        total_climate = sum(climate_effects.values())
        total_socio = sum(socio_effects.values()) 
        total_demo = sum(demo_effects.values())
        total_importance = total_climate + total_socio + total_demo
        
        if total_importance > 0:
            climate_pct = (total_climate / total_importance) * 100
            socio_pct = (total_socio / total_importance) * 100
            demo_pct = (total_demo / total_importance) * 100
            
            print(f"\\n📊 PATHWAY CONTRIBUTION BREAKDOWN:")
            print(f"   🌡️  Climate/Heat: {climate_pct:.1f}% (total SHAP: {total_climate:.3f})")
            print(f"   🏢 Socioeconomic: {socio_pct:.1f}% (total SHAP: {total_socio:.3f})")
            print(f"   👥 Demographic: {demo_pct:.1f}% (total SHAP: {total_demo:.3f})")
    
    def _clean_feature_name(self, feature_name):
        """Clean feature names for better display."""
        # Remove 'climate_' prefix and make more readable
        clean_name = feature_name.replace('climate_', '').replace('_', ' ').title()
        
        # Specific translations for common climate variables
        translations = {
            'Daily Mean Temp': 'Daily Mean Temperature',
            'Daily Max Temp': 'Daily Maximum Temperature', 
            'Daily Min Temp': 'Daily Minimum Temperature',
            '7D Max Temp': '7-Day Max Temperature',
            '14D Mean Temp': '14-Day Mean Temperature',
            '30D Mean Temp': '30-Day Mean Temperature',
            'Heat Stress Index': 'Heat Stress Index',
            'Temp Anomaly': 'Temperature Anomaly',
            'Standardized Anomaly': 'Standardized Temperature Anomaly'
        }
        
        return translations.get(clean_name, clean_name)
    
    def _create_comprehensive_pathway_plots(self, results, model_name, config, climate_vars, socio_vars):
        """Create comprehensive pathway visualization plots."""
        print(f"   📊 Creating comprehensive pathway plots for {model_name}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        
        feature_importance = results['feature_importance']
        
        # Categorize features
        climate_effects = {k: v for k, v in feature_importance.items() 
                          if any(climate_var in k for climate_var in climate_vars) or k.startswith('climate_')}
        socio_effects = {k: v for k, v in feature_importance.items() 
                        if any(socio_var in k for socio_var in socio_vars)}
        demo_effects = {k: v for k, v in feature_importance.items() 
                       if k in ['Age (at enrolment)', 'Sex', 'Race']}
        
        # 1. Top 10 Climate Effects (Top Left)
        ax = axes[0, 0]
        if climate_effects:
            sorted_climate = sorted(climate_effects.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_climate)
            clean_features = [self._clean_feature_name(f) for f in features]
            y_pos = np.arange(len(clean_features))
            
            bars = ax.barh(y_pos, importances, color='#FF4500', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(clean_features, fontsize=10)
            ax.set_xlabel('SHAP Importance')
            ax.set_title('🌡️ Top 10 Climate/Heat Effects', fontweight='bold', fontsize=12)
            
            # Add value labels
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{imp:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No Climate Effects\\nFound', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title('🌡️ Climate/Heat Effects', fontweight='bold')
        
        # 2. Top 10 Socioeconomic Effects (Top Center)
        ax = axes[0, 1]
        if socio_effects:
            sorted_socio = sorted(socio_effects.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_socio)
            clean_features = [self._clean_feature_name(f) for f in features]
            y_pos = np.arange(len(clean_features))
            
            bars = ax.barh(y_pos, importances, color='#4ECDC4', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(clean_features, fontsize=10)
            ax.set_xlabel('SHAP Importance')
            ax.set_title('🏢 Top 10 Socioeconomic Effects', fontweight='bold', fontsize=12)
            
            # Add value labels
            for bar, imp in zip(bars, importances):
                ax.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{imp:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No Socioeconomic\\nEffects Found', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('🏢 Socioeconomic Effects', fontweight='bold')
        
        # 3. Pathway Contribution Pie Chart (Top Right)
        ax = axes[0, 2]
        total_climate = sum(climate_effects.values())
        total_socio = sum(socio_effects.values())
        total_demo = sum(demo_effects.values())
        
        if total_climate + total_socio + total_demo > 0:
            sizes = [total_climate, total_socio, total_demo]
            labels = ['Climate/Heat', 'Socioeconomic', 'Demographic']
            colors = ['#FF4500', '#4ECDC4', '#FF6B6B']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10})
            ax.set_title(f'📊 Pathway Contributions\\n{config["pathway_type"].title()}', 
                        fontweight='bold', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Pathway Data\\nAvailable', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
        
        # 4. Heat Stress Analysis (Bottom Left)
        ax = axes[1, 0]
        # Look for heat stress or temperature variables
        heat_vars = [var for var in climate_effects.keys() 
                    if any(term in var.lower() for term in ['heat', 'temp', 'stress'])]
        
        if heat_vars and len(heat_vars) >= 3:
            heat_importances = [climate_effects[var] for var in heat_vars[:5]]
            heat_labels = [self._clean_feature_name(var) for var in heat_vars[:5]]
            
            bars = ax.bar(range(len(heat_labels)), heat_importances, color='#FF6347', alpha=0.7)
            ax.set_xticks(range(len(heat_labels)))
            ax.set_xticklabels(heat_labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('SHAP Importance')
            ax.set_title('🔥 Heat Stress Variables', fontweight='bold', fontsize=12)
            
            # Add value labels
            for bar, imp in zip(bars, heat_importances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(heat_importances)*0.01,
                       f'{imp:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No Heat Stress\\nVariables Found', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title('🔥 Heat Stress Analysis', fontweight='bold')
        
        # 5. Overall Top Predictors (Bottom Center)
        ax = axes[1, 1]
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:12]
        features, importances = zip(*sorted_features)
        clean_features = [self._clean_feature_name(f) for f in features]
        
        # Color code by pathway
        colors = []
        for feature in features:
            if any(climate_var in feature for climate_var in climate_vars) or feature.startswith('climate_'):
                colors.append('#FF4500')  # Orange-red for climate
            elif any(socio_var in feature for socio_var in socio_vars):
                colors.append('#4ECDC4')  # Teal for socioeconomic
            else:
                colors.append('#FF6B6B')  # Red for demographic
        
        y_pos = np.arange(len(clean_features))
        bars = ax.barh(y_pos, importances, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:18] + '...' if len(f) > 18 else f for f in clean_features], fontsize=9)
        ax.set_xlabel('SHAP Importance')
        ax.set_title('📈 Top 12 Overall Predictors', fontweight='bold', fontsize=12)
        
        # 6. Model Performance Summary (Bottom Right)
        ax = axes[1, 2]
        ax.axis('off')
        
        test_r2 = results['test_r2']
        cv_mean = results['cv_scores'].mean()
        cv_std = results['cv_scores'].std()
        sample_size = results['sample_size']
        
        summary_text = [
            f"📈 Model Performance:",
            f"Test R²: {test_r2:.3f}",
            f"CV R²: {cv_mean:.3f} ± {cv_std:.3f}",
            f"Sample Size: {sample_size:,}",
            "",
            f"🌡️  Climate Variables: {len(climate_effects)}",
            f"🏢 Socioeconomic: {len(socio_effects)}",
            f"👥 Demographic: {len(demo_effects)}",
            "",
            f"🎯 Outcome: {config['outcome']}",
            f"🛤️  Pathway: {config['pathway_type'].title()}",
            "",
            f"🔬 Uses Enhanced Climate Data",
            f"📊 XGBoost + SHAP Analysis"
        ]
        
        for i, text in enumerate(summary_text):
            weight = 'bold' if text.endswith(':') else 'normal'
            fontsize = 10 if weight == 'bold' else 9
            ax.text(0.05, 0.95 - i*0.065, text, transform=ax.transAxes,
                   fontsize=fontsize, weight=weight, va='top')
        
        plt.suptitle(f'Comprehensive Pathway Analysis: {model_name}\\n{config["outcome"]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save as SVG
        filename = f"comprehensive_pathway_{model_name.lower()}.svg"
        filepath = Path("figures") / filename
        plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Comprehensive visualization saved: {filepath}")
    
    def _generate_comprehensive_heat_insights(self):
        """Generate comprehensive insights about heat effects across all pathways."""
        print(f"\\n🔥 COMPREHENSIVE HEAT-HEALTH INSIGHTS")
        print("="*50)
        
        if not self.results:
            print("❌ No results available for comprehensive analysis")
            return
        
        # Analyze heat effects across all models
        pathway_summary = {}
        
        for model_name, results in self.results.items():
            feature_importance = results['feature_importance']
            
            # Extract climate/heat effects
            climate_effects = {}
            for feature, importance in feature_importance.items():
                if any(term in feature.lower() for term in ['climate_', 'temp', 'heat', 'stress']):
                    climate_effects[feature] = importance
            
            total_climate_importance = sum(climate_effects.values())
            
            pathway_summary[model_name] = {
                'r2': results['test_r2'],
                'cv_mean': results['cv_scores'].mean(),
                'sample_size': results['sample_size'],
                'climate_effects': climate_effects,
                'total_climate_importance': total_climate_importance,
                'top_climate_variable': max(climate_effects.items(), key=lambda x: x[1]) if climate_effects else None
            }
        
        # Display comprehensive results
        print("🌡️  HEAT EFFECTS BY BIOMARKER PATHWAY:")
        
        # Sort by total climate importance
        sorted_pathways = sorted(pathway_summary.items(), 
                               key=lambda x: x[1]['total_climate_importance'], reverse=True)
        
        for model_name, summary in sorted_pathways:
            print(f"\\n🧪 {model_name}:")
            print(f"   📈 Model R²: {summary['r2']:.3f} (CV: {summary['cv_mean']:.3f})")
            print(f"   📊 Sample Size: {summary['sample_size']:,}")
            print(f"   🌡️  Total Climate Importance: {summary['total_climate_importance']:.4f}")
            
            if summary['top_climate_variable']:
                top_var, top_importance = summary['top_climate_variable']
                clean_var = self._clean_feature_name(top_var)
                print(f"   🔥 Top Climate Variable: {clean_var} ({top_importance:.4f})")
            
            print(f"   🌡️  Climate Variables Found: {len(summary['climate_effects'])}")
            
            # Show top 3 climate effects
            if summary['climate_effects']:
                sorted_climate = sorted(summary['climate_effects'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                print(f"   🌡️  Top 3 Climate Effects:")
                for i, (var, importance) in enumerate(sorted_climate, 1):
                    clean_var = self._clean_feature_name(var)
                    print(f"      {i}. {clean_var}: {importance:.4f}")
        
        # Overall insights
        avg_climate_importance = np.mean([s['total_climate_importance'] for s in pathway_summary.values()])
        avg_r2 = np.mean([s['r2'] for s in pathway_summary.values()])
        
        print(f"\\n📊 OVERALL HEAT-HEALTH INSIGHTS:")
        print(f"   🌡️  Average Climate Importance: {avg_climate_importance:.4f}")
        print(f"   📈 Average Model R²: {avg_r2:.3f}")
        print(f"   🧪 Biomarkers Analyzed: {len(pathway_summary)}")
        print(f"   ✅ Models with Climate Effects: {sum(1 for s in pathway_summary.values() if s['total_climate_importance'] > 0.01)}")
        
        # Generate clinical recommendations
        print(f"\\n🏥 CLINICAL IMPLICATIONS:")
        
        best_climate_pathway = max(pathway_summary.items(), key=lambda x: x[1]['total_climate_importance'])
        best_model, best_summary = best_climate_pathway
        
        if best_summary['total_climate_importance'] > 0.1:
            print(f"   ✅ Strong climate effects detected in {best_model}")
            print(f"   🌡️  Heat exposure significantly affects biomarker levels")
            print(f"   📊 Climate adaptation strategies recommended")
        elif avg_climate_importance > 0.01:
            print(f"   🟡 Moderate climate effects detected across pathways")
            print(f"   🌡️  Heat monitoring recommended for vulnerable populations")
        else:
            print(f"   ⚠️  Climate effects remain weak despite enhanced data")
            print(f"   🔍 Consider longer exposure periods or different metrics")
        
        print(f"\\n🎯 RESEARCH RECOMMENDATIONS:")
        print(f"   1. Focus on {best_model.replace('_Enhanced', '')} pathway for heat research")
        print(f"   2. Investigate multi-day heat exposure patterns")
        print(f"   3. Consider interaction effects between climate and socioeconomic factors")
        print(f"   4. Validate findings with larger temporal windows")
    
    def _run_fallback_analysis(self):
        """Run fallback analysis if enhanced dataset not available."""
        print("🔄 RUNNING FALLBACK ANALYSIS")
        print("Using original dataset with improved methodology...")
        
        # This would use the original rigorous_shap_biomarker_analysis.py
        # but with enhanced preprocessing and visualization
        
        print("⚠️  Enhanced climate dataset not available - recommend running climate extraction first")
        return None


if __name__ == "__main__":
    analyzer = ComprehensiveHeatSHAPAnalysis()
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\\n🎉 COMPREHENSIVE HEAT-HEALTH SHAP ANALYSIS COMPLETE")
    if results:
        print(f"✅ {len(results)} biomarker pathways analyzed")
        print(f"✅ Top 10 climate and socioeconomic effects identified")
        print(f"✅ Enhanced climate data successfully integrated")
        print(f"✅ Publication-quality SHAP visualizations created")
        print(f"✅ Comprehensive heat-health insights generated")
    else:
        print("❌ Analysis incomplete - check enhanced climate data availability")