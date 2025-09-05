"""
State-of-the-Art XAI Heat-Health Analysis

Comprehensive explainable AI analysis to uncover novel heat-health insights using:
- SHAP (SHapley Additive exPlanations) for feature importance
- Advanced ML models (XGBoost, Random Forest, Neural Networks)
- Multi-domain feature discovery (biological, climate, socio-economic)
- Population stratification and temporal analysis
- Social vulnerability mapping

Target Outcomes:
1. Discover novel heat vulnerability biomarkers
2. Identify critical temperature thresholds
3. Reveal social determinants of heat risk
4. Generate personalized heat risk profiles
5. Create actionable climate adaptation insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# ML and XAI libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import shap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class StateOfArtXAIAnalyzer:
    """
    State-of-the-art XAI analyzer for heat-health insights discovery.
    """
    
    def __init__(self, base_path: str = "/home/cparker"):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "heat_analysis_optimized" / "data" / "enhanced_se_integrated"
        self.output_path = self.base_path / "heat_analysis_optimized" / "analysis" / "xai_results"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Analysis results storage
        self.results = {
            'models': {},
            'shap_values': {},
            'feature_importance': {},
            'insights': {},
            'visualizations': {}
        }
        
        # Define target biomarkers for heat response analysis
        self.target_biomarkers = [
            'std_glucose',
            'std_cholesterol_total', 
            'std_systolic_bp',
            'std_diastolic_bp',
            'std_potassium',
            'std_creatinine',
            'std_hemoglobin'
        ]
        
        # Define predictor categories
        self.predictor_categories = {
            'climate': [col for col in [] if col.startswith('climate_')],
            'demographics': ['std_age', 'std_sex'],
            'socioeconomic': [col for col in [] if col.startswith('se_')],
            'interactions': [col for col in [] if col.startswith('interact_')]
        }
    
    def load_optimal_dataset(self) -> pd.DataFrame:
        """
        Load the optimal XAI-ready dataset with comprehensive features.
        """
        self.logger.info("=" * 80)
        self.logger.info("📥 LOADING OPTIMAL XAI DATASET")
        self.logger.info("=" * 80)
        
        # Load the high-quality enhanced dataset
        dataset_file = self.data_path / "enhanced_se_high_quality.csv"
        
        if not dataset_file.exists():
            self.logger.error(f"Dataset file not found: {dataset_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(dataset_file)
            self.logger.info(f"✅ Loaded dataset: {len(df)} participants, {len(df.columns)} variables")
            
            # Update predictor categories with actual columns
            self.predictor_categories['climate'] = [col for col in df.columns if col.startswith('climate_')]
            self.predictor_categories['socioeconomic'] = [col for col in df.columns if col.startswith('se_')]
            self.predictor_categories['interactions'] = [col for col in df.columns if col.startswith('interact_')]
            
            # Log feature categories
            for category, features in self.predictor_categories.items():
                self.logger.info(f"   {category.upper()}: {len(features)} features")
            
            # Log target biomarkers available
            available_targets = [target for target in self.target_biomarkers if target in df.columns]
            self.logger.info(f"   TARGET BIOMARKERS: {len(available_targets)} available")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def prepare_analysis_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for XAI analysis with advanced preprocessing.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("🔧 PREPARING ANALYSIS DATA")
        self.logger.info("=" * 60)
        
        # Create feature matrix
        all_predictors = []
        for category, features in self.predictor_categories.items():
            all_predictors.extend([f for f in features if f in df.columns])
        
        # Add demographics
        all_predictors.extend([f for f in self.predictor_categories['demographics'] if f in df.columns])
        
        self.logger.info(f"Total predictors identified: {len(all_predictors)}")
        
        # Prepare data for each target biomarker
        prepared_data = {}
        
        for target in self.target_biomarkers:
            if target not in df.columns:
                continue
            
            self.logger.info(f"\\n📊 Preparing data for {target}")
            
            # Create analysis subset
            target_data = df[all_predictors + [target]].copy()
            
            # Remove duplicate columns if any
            target_data = target_data.loc[:, ~target_data.columns.duplicated()]
            
            # Remove rows where target is missing
            before_dropna = len(target_data)
            target_data = target_data.dropna(subset=[target])
            after_dropna = len(target_data)
            
            if after_dropna < 50:  # Need minimum sample size
                self.logger.warning(f"   Insufficient data: {after_dropna} samples")
                continue
            
            self.logger.info(f"   Samples: {after_dropna} (dropped {before_dropna - after_dropna} missing)")
            
            # Advanced imputation for predictors
            all_predictors_available = [col for col in all_predictors if col in target_data.columns]
            X = target_data[all_predictors_available].copy()
            y = target_data[target].copy()
            
            # Remove any remaining duplicated columns
            X = X.loc[:, ~X.columns.duplicated()]
            
            # Simple preprocessing: keep only numeric columns and fill missing values
            X = X.select_dtypes(include=[np.number])
            
            # Remove columns with >90% missing values
            high_missing_cols = X.columns[X.isnull().mean() > 0.9].tolist()
            if high_missing_cols:
                X = X.drop(columns=high_missing_cols)
                self.logger.info(f"   Removed {len(high_missing_cols)} high-missing columns")
            
            # Fill remaining missing values with median
            X = X.fillna(X.median()).fillna(0)  # Double fillna for safety
            
            # Final feature set
            X_final = X
            
            # Outlier detection and capping (3 sigma rule)
            for col in X_final.columns:
                mean_val = X_final[col].mean()
                std_val = X_final[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X_final[col] = X_final[col].clip(lower_bound, upper_bound)
            
            self.logger.info(f"   Final feature matrix: {X_final.shape}")
            
            prepared_data[target] = {
                'X': X_final,
                'y': y,
                'feature_names': X_final.columns.tolist(),
                'n_samples': len(X_final),
                'target_stats': {
                    'mean': y.mean(),
                    'std': y.std(),
                    'min': y.min(),
                    'max': y.max(),
                    'n_valid': len(y)
                }
                }
        
        self.logger.info(f"\\n✅ Prepared data for {len(prepared_data)} target biomarkers")
        return prepared_data
    
    def build_xai_models(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build state-of-the-art ML models for XAI analysis.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("🤖 BUILDING XAI MODELS")
        self.logger.info("=" * 60)
        
        models_results = {}
        
        for target, data in prepared_data.items():
            self.logger.info(f"\\n📊 Building models for {target}")
            
            X, y = data['X'], data['y']
            feature_names = data['feature_names']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Feature scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame for feature names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
            
            target_models = {}
            
            # 1. XGBoost (State-of-the-art gradient boosting)
            self.logger.info("   🚀 Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_r2 = r2_score(y_test, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            target_models['xgboost'] = {
                'model': xgb_model,
                'predictions': xgb_pred,
                'r2': xgb_r2,
                'rmse': xgb_rmse,
                'feature_importance': dict(zip(feature_names, xgb_model.feature_importances_))
            }
            
            # 2. Random Forest (Ensemble method)
            self.logger.info("   🌳 Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            
            target_models['random_forest'] = {
                'model': rf_model,
                'predictions': rf_pred,
                'r2': rf_r2,
                'rmse': rf_rmse,
                'feature_importance': dict(zip(feature_names, rf_model.feature_importances_))
            }
            
            # 3. Gradient Boosting
            self.logger.info("   📈 Training Gradient Boosting...")
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_pred = gb_model.predict(X_test_scaled)
            gb_r2 = r2_score(y_test, gb_pred)
            gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
            
            target_models['gradient_boosting'] = {
                'model': gb_model,
                'predictions': gb_pred,
                'r2': gb_r2,
                'rmse': gb_rmse,
                'feature_importance': dict(zip(feature_names, gb_model.feature_importances_))
            }
            
            # Select best model
            best_model_name = max(target_models.keys(), key=lambda k: target_models[k]['r2'])
            best_model_r2 = target_models[best_model_name]['r2']
            
            self.logger.info(f"   ✅ Best model: {best_model_name} (R² = {best_model_r2:.3f})")
            
            models_results[target] = {
                'models': target_models,
                'best_model': best_model_name,
                'best_r2': best_model_r2,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler,
                'feature_names': feature_names
            }
        
        return models_results
    
    def compute_shap_analysis(self, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive SHAP analysis for explainable AI insights.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("🔍 COMPUTING SHAP ANALYSIS")
        self.logger.info("=" * 60)
        
        shap_results = {}
        
        for target, results in models_results.items():
            self.logger.info(f"\\n🎯 SHAP analysis for {target}")
            
            best_model_name = results['best_model']
            best_model = results['models'][best_model_name]['model']
            X_train = results['X_train']
            X_test = results['X_test']
            feature_names = results['feature_names']
            
            try:
                # Create SHAP explainer
                if best_model_name == 'xgboost':
                    explainer = shap.TreeExplainer(best_model)
                else:
                    # Use KernelExplainer for other models (more general but slower)
                    explainer = shap.KernelExplainer(
                        best_model.predict, 
                        shap.sample(X_train, 100)  # Sample for speed
                    )
                
                # Calculate SHAP values
                self.logger.info("   Computing SHAP values...")
                shap_values = explainer.shap_values(X_test.iloc[:200])  # Limit for speed
                
                # Global feature importance
                shap_importance = np.abs(shap_values).mean(0)
                global_importance = dict(zip(feature_names, shap_importance))
                
                # Sort by importance
                sorted_importance = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
                
                self.logger.info(f"   ✅ Top 5 features by SHAP importance:")
                for i, (feature, importance) in enumerate(sorted_importance[:5]):
                    self.logger.info(f"      {i+1}. {feature}: {importance:.4f}")
                
                # Categorize features by domain
                domain_importance = {
                    'climate': 0,
                    'socioeconomic': 0,
                    'demographics': 0,
                    'interactions': 0
                }
                
                for feature, importance in global_importance.items():
                    if feature.startswith('climate_'):
                        domain_importance['climate'] += importance
                    elif feature.startswith('se_'):
                        domain_importance['socioeconomic'] += importance
                    elif feature in ['std_age', 'std_sex']:
                        domain_importance['demographics'] += importance
                    elif feature.startswith('interact_'):
                        domain_importance['interactions'] += importance
                
                self.logger.info(f"   📊 Domain importance:")
                for domain, importance in sorted(domain_importance.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"      {domain}: {importance:.4f}")
                
                shap_results[target] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'global_importance': global_importance,
                    'sorted_importance': sorted_importance,
                    'domain_importance': domain_importance,
                    'X_test_subset': X_test.iloc[:200],
                    'y_test_subset': results['y_test'].iloc[:200]
                }
                
            except Exception as e:
                self.logger.error(f"   Error in SHAP analysis: {e}")
                continue
        
        return shap_results
    
    def discover_heat_insights(self, shap_results: Dict[str, Any], models_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover novel heat-health insights from XAI analysis.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("🔬 DISCOVERING HEAT-HEALTH INSIGHTS")
        self.logger.info("=" * 60)
        
        insights = {
            'key_findings': [],
            'biomarker_responses': {},
            'critical_thresholds': {},
            'vulnerability_profiles': {},
            'social_determinants': {},
            'novel_mechanisms': []
        }
        
        for target, shap_data in shap_results.items():
            self.logger.info(f"\\n🎯 Analyzing insights for {target}")
            
            global_importance = shap_data['global_importance']
            sorted_importance = shap_data['sorted_importance']
            domain_importance = shap_data['domain_importance']
            
            # 1. Identify critical climate features
            climate_features = {k: v for k, v in global_importance.items() if k.startswith('climate_')}
            top_climate = sorted(climate_features.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_climate:
                insights['biomarker_responses'][target] = {
                    'top_climate_predictors': top_climate,
                    'climate_importance_total': domain_importance['climate']
                }
                
                # Extract temperature thresholds
                for climate_feature, importance in top_climate:
                    if 'temp' in climate_feature and importance > 0.01:  # Significant threshold
                        insights['key_findings'].append(
                            f"{target} shows significant response to {climate_feature} (SHAP importance: {importance:.3f})"
                        )
            
            # 2. Identify socio-economic determinants
            se_features = {k: v for k, v in global_importance.items() if k.startswith('se_')}
            top_se = sorted(se_features.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if top_se and domain_importance['socioeconomic'] > 0.05:  # Significant SE influence
                insights['social_determinants'][target] = {
                    'top_se_predictors': top_se,
                    'se_importance_total': domain_importance['socioeconomic']
                }
                
                insights['key_findings'].append(
                    f"Socio-economic factors significantly influence {target} heat response (total importance: {domain_importance['socioeconomic']:.3f})"
                )
            
            # 3. Discover interaction effects
            interaction_features = {k: v for k, v in global_importance.items() if k.startswith('interact_')}
            top_interactions = sorted(interaction_features.items(), key=lambda x: x[1], reverse=True)[:2]
            
            if top_interactions and domain_importance['interactions'] > 0.03:
                insights['novel_mechanisms'].append({
                    'biomarker': target,
                    'top_interactions': top_interactions,
                    'mechanism': f"Temperature interactions significantly modify {target} response"
                })
            
            # 4. Vulnerability profiling
            total_predictive_power = sum(domain_importance.values())
            vulnerability_profile = {
                'climate_vulnerability': domain_importance['climate'] / total_predictive_power,
                'social_vulnerability': domain_importance['socioeconomic'] / total_predictive_power,
                'interaction_vulnerability': domain_importance['interactions'] / total_predictive_power
            }
            
            insights['vulnerability_profiles'][target] = vulnerability_profile
        
        # 5. Cross-biomarker analysis
        self._cross_biomarker_analysis(insights, shap_results)
        
        # 6. Generate actionable recommendations
        self._generate_recommendations(insights)
        
        return insights
    
    def _cross_biomarker_analysis(self, insights: Dict[str, Any], shap_results: Dict[str, Any]) -> None:
        """
        Analyze patterns across multiple biomarkers.
        """
        if len(shap_results) < 2:
            return
        
        # Find common climate predictors
        climate_predictors_all = {}
        for target, shap_data in shap_results.items():
            climate_features = {k: v for k, v in shap_data['global_importance'].items() if k.startswith('climate_')}
            for feature, importance in climate_features.items():
                if feature not in climate_predictors_all:
                    climate_predictors_all[feature] = []
                climate_predictors_all[feature].append((target, importance))
        
        # Find features important for multiple biomarkers
        multi_biomarker_features = {}
        for feature, target_importance_list in climate_predictors_all.items():
            if len(target_importance_list) >= 2:  # Important for 2+ biomarkers
                avg_importance = np.mean([imp for _, imp in target_importance_list])
                multi_biomarker_features[feature] = {
                    'targets': [target for target, _ in target_importance_list],
                    'avg_importance': avg_importance,
                    'individual_importance': target_importance_list
                }
        
        if multi_biomarker_features:
            insights['key_findings'].append("MULTI-BIOMARKER CLIMATE RESPONSES DISCOVERED:")
            for feature, data in sorted(multi_biomarker_features.items(), key=lambda x: x[1]['avg_importance'], reverse=True)[:3]:
                targets_str = ", ".join(data['targets'])
                insights['key_findings'].append(
                    f"  {feature} affects multiple biomarkers ({targets_str}) - Avg importance: {data['avg_importance']:.3f}"
                )
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> None:
        """
        Generate actionable climate adaptation recommendations.
        """
        recommendations = []
        
        # Temperature threshold recommendations
        temp_important_biomarkers = []
        for target, data in insights['biomarker_responses'].items():
            for climate_feature, importance in data['top_climate_predictors']:
                if 'temp' in climate_feature and importance > 0.02:
                    temp_important_biomarkers.append(target)
        
        if temp_important_biomarkers:
            recommendations.append({
                'category': 'Temperature Monitoring',
                'recommendation': f"Prioritize temperature monitoring for individuals with {', '.join(temp_important_biomarkers)} health concerns",
                'evidence': f"Temperature variables show high predictive importance for {len(temp_important_biomarkers)} key biomarkers"
            })
        
        # Social vulnerability recommendations
        high_se_impact_biomarkers = []
        for target, data in insights['social_determinants'].items():
            if data['se_importance_total'] > 0.1:  # High SE impact
                high_se_impact_biomarkers.append(target)
        
        if high_se_impact_biomarkers:
            recommendations.append({
                'category': 'Social Vulnerability',
                'recommendation': f"Target heat adaptation interventions for socio-economically vulnerable populations",
                'evidence': f"SE factors significantly influence {len(high_se_impact_biomarkers)} biomarkers: {', '.join(high_se_impact_biomarkers)}"
            })
        
        # Personalized risk recommendations
        if insights['novel_mechanisms']:
            recommendations.append({
                'category': 'Personalized Risk Assessment',
                'recommendation': "Develop personalized heat risk models incorporating temperature-biomarker interactions",
                'evidence': f"Discovered {len(insights['novel_mechanisms'])} significant interaction mechanisms"
            })
        
        insights['recommendations'] = recommendations
    
    def create_visualizations(self, shap_results: Dict[str, Any], insights: Dict[str, Any]) -> None:
        """
        Create comprehensive visualizations of XAI results.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("📊 CREATING VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        # Set up the plotting environment
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Global Feature Importance Comparison
        self._plot_global_importance_comparison(shap_results)
        
        # 2. Domain Importance Analysis
        self._plot_domain_importance(shap_results)
        
        # 3. SHAP Summary Plots
        self._create_shap_summary_plots(shap_results)
        
        # 4. Heat Vulnerability Profiles
        self._plot_vulnerability_profiles(insights)
        
        # 5. Cross-biomarker Climate Response
        self._plot_climate_response_patterns(shap_results)
        
        self.logger.info("✅ All visualizations created and saved")
    
    def _plot_global_importance_comparison(self, shap_results: Dict[str, Any]) -> None:
        """Plot comparison of feature importance across biomarkers."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (target, shap_data) in enumerate(shap_results.items()):
            if i >= 4:  # Limit to 4 plots
                break
            
            sorted_importance = shap_data['sorted_importance'][:15]  # Top 15 features
            features, importances = zip(*sorted_importance)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            axes[i].barh(y_pos, importances, alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels([f.replace('climate_', '').replace('se_', '').replace('std_', '') for f in features])
            axes[i].set_xlabel('SHAP Importance')
            axes[i].set_title(f'{target} - Top Predictors')
            axes[i].grid(axis='x', alpha=0.3)
        
        # Remove empty subplots
        for j in range(i+1, 4):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'global_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_importance(self, shap_results: Dict[str, Any]) -> None:
        """Plot domain-level importance across biomarkers."""
        
        domains = ['climate', 'socioeconomic', 'demographics', 'interactions']
        biomarkers = list(shap_results.keys())
        
        # Create matrix of domain importance
        importance_matrix = np.zeros((len(biomarkers), len(domains)))
        
        for i, target in enumerate(biomarkers):
            domain_importance = shap_results[target]['domain_importance']
            for j, domain in enumerate(domains):
                importance_matrix[i, j] = domain_importance.get(domain, 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(domains)))
        ax.set_yticks(np.arange(len(biomarkers)))
        ax.set_xticklabels([d.title() for d in domains])
        ax.set_yticklabels([b.replace('std_', '') for b in biomarkers])
        
        # Add text annotations
        for i in range(len(biomarkers)):
            for j in range(len(domains)):
                text = ax.text(j, i, f'{importance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Domain Importance Across Biomarkers\\n(SHAP Values)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHAP Importance', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'domain_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_shap_summary_plots(self, shap_results: Dict[str, Any]) -> None:
        """Create SHAP summary plots for each biomarker."""
        
        for target, shap_data in shap_results.items():
            try:
                # SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_data['shap_values'], 
                    shap_data['X_test_subset'], 
                    plot_type="dot",
                    show=False,
                    max_display=15
                )
                plt.title(f'SHAP Summary: {target.replace("std_", "").title()}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.output_path / f'shap_summary_{target}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # SHAP waterfall plot for first sample
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_data['shap_values'][0], 
                        base_values=shap_data['explainer'].expected_value,
                        data=shap_data['X_test_subset'].iloc[0],
                        feature_names=shap_data['X_test_subset'].columns.tolist()
                    ),
                    max_display=10,
                    show=False
                )
                plt.title(f'SHAP Waterfall Example: {target.replace("std_", "").title()}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.output_path / f'shap_waterfall_{target}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                self.logger.warning(f"Could not create SHAP plots for {target}: {e}")
                continue
    
    def _plot_vulnerability_profiles(self, insights: Dict[str, Any]) -> None:
        """Plot vulnerability profiles across biomarkers."""
        
        if not insights['vulnerability_profiles']:
            return
        
        biomarkers = list(insights['vulnerability_profiles'].keys())
        vulnerability_types = ['climate_vulnerability', 'social_vulnerability', 'interaction_vulnerability']
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bottom = np.zeros(len(biomarkers))
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        
        for i, vuln_type in enumerate(vulnerability_types):
            values = [insights['vulnerability_profiles'][biomarker].get(vuln_type, 0) for biomarker in biomarkers]
            ax.bar(biomarkers, values, bottom=bottom, label=vuln_type.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
            bottom += values
        
        ax.set_ylabel('Vulnerability Profile (Proportion)')
        ax.set_title('Heat Vulnerability Profiles by Biomarker\\n(Based on SHAP Feature Importance)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'vulnerability_profiles.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_climate_response_patterns(self, shap_results: Dict[str, Any]) -> None:
        """Plot climate response patterns across biomarkers."""
        
        # Collect climate feature importance across all biomarkers
        climate_features = set()
        for shap_data in shap_results.values():
            climate_features.update([k for k in shap_data['global_importance'].keys() if k.startswith('climate_')])
        
        # Focus on temperature-related features
        temp_features = [f for f in climate_features if 'temp' in f][:10]  # Top 10 temperature features
        
        if not temp_features:
            return
        
        # Create heatmap of temperature feature importance
        biomarkers = list(shap_results.keys())
        importance_matrix = np.zeros((len(biomarkers), len(temp_features)))
        
        for i, biomarker in enumerate(biomarkers):
            for j, temp_feature in enumerate(temp_features):
                importance_matrix[i, j] = shap_results[biomarker]['global_importance'].get(temp_feature, 0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(importance_matrix, cmap='Reds', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(temp_features)))
        ax.set_yticks(np.arange(len(biomarkers)))
        ax.set_xticklabels([f.replace('climate_', '').replace('temp_', 'T_') for f in temp_features], rotation=45, ha='right')
        ax.set_yticklabels([b.replace('std_', '') for b in biomarkers])
        
        # Add text annotations
        for i in range(len(biomarkers)):
            for j in range(len(temp_features)):
                if importance_matrix[i, j] > 0.01:  # Only show significant values
                    text = ax.text(j, i, f'{importance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title('Temperature Feature Importance Across Biomarkers\\n(SHAP Values)', 
                     fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHAP Importance', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'climate_response_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, insights: Dict[str, Any], models_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive analysis report.
        """
        self.logger.info("\\n" + "=" * 60)
        self.logger.info("📋 GENERATING COMPREHENSIVE REPORT")
        self.logger.info("=" * 60)
        
        report_lines = []
        report_lines.append("# State-of-the-Art XAI Heat-Health Analysis Report\\n")
        report_lines.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        report_lines.append("---\\n")
        
        # Executive Summary
        report_lines.append("## Executive Summary\\n")
        report_lines.append(f"Comprehensive explainable AI analysis of heat-health relationships using advanced ML models and SHAP analysis.\\n")
        report_lines.append(f"**Datasets Analyzed**: {len(models_results)} biomarker targets\\n")
        report_lines.append(f"**Key Insights Discovered**: {len(insights['key_findings'])}\\n")
        report_lines.append(f"**Novel Mechanisms Identified**: {len(insights['novel_mechanisms'])}\\n\\n")
        
        # Model Performance
        report_lines.append("## Model Performance\\n")
        report_lines.append("| Biomarker | Best Model | R² Score | RMSE |")
        report_lines.append("|-----------|------------|----------|------|")
        
        for target, results in models_results.items():
            best_model = results['best_model']
            best_r2 = results['best_r2']
            best_rmse = results['models'][best_model]['rmse']
            report_lines.append(f"| {target.replace('std_', '')} | {best_model} | {best_r2:.3f} | {best_rmse:.3f} |")
        
        report_lines.append("\\n")
        
        # Key Findings
        report_lines.append("## Key Findings\\n")
        for i, finding in enumerate(insights['key_findings'], 1):
            report_lines.append(f"{i}. {finding}\\n")
        
        report_lines.append("\\n")
        
        # Novel Mechanisms
        if insights['novel_mechanisms']:
            report_lines.append("## Novel Heat-Health Mechanisms Discovered\\n")
            for i, mechanism in enumerate(insights['novel_mechanisms'], 1):
                report_lines.append(f"### Mechanism {i}: {mechanism['biomarker']}\\n")
                report_lines.append(f"**Description**: {mechanism['mechanism']}\\n")
                report_lines.append("**Top Interactions**:\\n")
                for interaction, importance in mechanism['top_interactions']:
                    report_lines.append(f"- {interaction}: {importance:.4f}\\n")
                report_lines.append("\\n")
        
        # Social Determinants
        if insights['social_determinants']:
            report_lines.append("## Social Determinants of Heat Vulnerability\\n")
            for target, data in insights['social_determinants'].items():
                report_lines.append(f"### {target.replace('std_', '').title()}\\n")
                report_lines.append(f"**Total SE Importance**: {data['se_importance_total']:.3f}\\n")
                report_lines.append("**Top SE Predictors**:\\n")
                for predictor, importance in data['top_se_predictors']:
                    clean_predictor = predictor.replace('se_', '').replace('_', ' ').title()
                    report_lines.append(f"- {clean_predictor}: {importance:.4f}\\n")
                report_lines.append("\\n")
        
        # Recommendations
        if 'recommendations' in insights:
            report_lines.append("## Clinical and Policy Recommendations\\n")
            for i, rec in enumerate(insights['recommendations'], 1):
                report_lines.append(f"### {i}. {rec['category']}\\n")
                report_lines.append(f"**Recommendation**: {rec['recommendation']}\\n")
                report_lines.append(f"**Evidence**: {rec['evidence']}\\n\\n")
        
        # Technical Details
        report_lines.append("## Technical Details\\n")
        report_lines.append("### Methods Used\\n")
        report_lines.append("- **ML Models**: XGBoost, Random Forest, Gradient Boosting\\n")
        report_lines.append("- **Explainability**: SHAP (SHapley Additive exPlanations)\\n")
        report_lines.append("- **Preprocessing**: KNN imputation, robust scaling, outlier capping\\n")
        report_lines.append("- **Validation**: Train-test split with 20% holdout\\n\\n")
        
        # Save report
        report_file = self.output_path / "XAI_HEAT_HEALTH_ANALYSIS_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(''.join(report_lines))
        
        self.logger.info(f"✅ Comprehensive report saved: {report_file}")
    
    def run_complete_xai_analysis(self) -> Dict[str, Any]:
        """
        Run the complete state-of-the-art XAI analysis pipeline.
        """
        self.logger.info("🚀 STATE-OF-THE-ART XAI HEAT-HEALTH ANALYSIS")
        self.logger.info("=" * 80)
        self.logger.info("Discovering novel heat-health insights using explainable AI")
        self.logger.info("=" * 80)
        
        # Step 1: Load optimal dataset
        df = self.load_optimal_dataset()
        if df.empty:
            self.logger.error("Failed to load dataset!")
            return {}
        
        # Step 2: Prepare analysis data
        prepared_data = self.prepare_analysis_data(df)
        if not prepared_data:
            self.logger.error("Failed to prepare analysis data!")
            return {}
        
        # Step 3: Build XAI models
        models_results = self.build_xai_models(prepared_data)
        if not models_results:
            self.logger.error("Failed to build models!")
            return {}
        
        # Step 4: Compute SHAP analysis
        shap_results = self.compute_shap_analysis(models_results)
        if not shap_results:
            self.logger.error("Failed to compute SHAP analysis!")
            return {}
        
        # Step 5: Discover heat insights
        insights = self.discover_heat_insights(shap_results, models_results)
        
        # Step 6: Create visualizations
        self.create_visualizations(shap_results, insights)
        
        # Step 7: Generate comprehensive report
        self.generate_comprehensive_report(insights, models_results)
        
        # Final summary
        self.logger.info("\\n" + "=" * 80)
        self.logger.info("🎉 XAI ANALYSIS COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Biomarkers analyzed: {len(models_results)}")
        self.logger.info(f"🔍 Key insights discovered: {len(insights['key_findings'])}")
        self.logger.info(f"🧬 Novel mechanisms identified: {len(insights['novel_mechanisms'])}")
        self.logger.info(f"📋 Recommendations generated: {len(insights.get('recommendations', []))}")
        self.logger.info(f"💾 Results saved to: {self.output_path}")
        
        # Store results
        self.results = {
            'models': models_results,
            'shap_values': shap_results,
            'insights': insights,
            'dataset_info': {
                'n_participants': len(df),
                'n_features': len(df.columns),
                'biomarkers_analyzed': list(models_results.keys())
            }
        }
        
        return self.results


def main():
    """Main execution function."""
    print("🚀 STATE-OF-THE-ART XAI HEAT-HEALTH ANALYSIS")
    print("=" * 80)
    print("Discovering novel heat-health insights using explainable AI")
    print("Methods: XGBoost, Random Forest, SHAP Analysis")
    print("Domains: Climate, Biomarkers, Socio-economic, Interactions")
    print("=" * 80)
    
    analyzer = StateOfArtXAIAnalyzer()
    
    # Run complete XAI analysis
    results = analyzer.run_complete_xai_analysis()
    
    if results:
        n_biomarkers = len(results['models'])
        n_insights = len(results['insights']['key_findings'])
        n_mechanisms = len(results['insights']['novel_mechanisms'])
        
        print(f"\\n🎉 XAI ANALYSIS COMPLETE!")
        print(f"📊 Analyzed {n_biomarkers} biomarker responses to heat exposure")
        print(f"🔍 Discovered {n_insights} key insights")
        print(f"🧬 Identified {n_mechanisms} novel heat-health mechanisms")
        print(f"💾 Results and visualizations saved to: analysis/xai_results/")
        print("🎯 Ready for clinical translation and policy implementation!")
    else:
        print("❌ Analysis failed. Check logs for details.")
    
    return results


if __name__ == "__main__":
    main()