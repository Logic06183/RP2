#!/usr/bin/env python
"""
Heat-Health XAI Framework - Core Analysis Pipeline

Scalable, dataset-agnostic pipeline for heat vulnerability analysis using explainable AI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import config
from ..data.loader import DataLoader
from ..features.climate import ClimateFeatureEngineer
from ..features.temporal import TemporalFeatureEngineer
from ..features.interactions import InteractionFeatureEngineer
from ..targets.pathways import PathwayTargetCreator
from ..models.optimization import ModelOptimizer
from ..explainability.shap_analysis import SHAPAnalyzer
from ..explainability.hypothesis import HypothesisGenerator

class HeatHealthAnalyzer:
    """Main analysis pipeline for heat-health XAI analysis."""
    
    def __init__(self, dataset_name: Optional[str] = None, 
                 custom_config: Optional[Dict] = None):
        """
        Initialize the analyzer.
        
        Args:
            dataset_name: Name of dataset configuration to load
            custom_config: Custom configuration overrides
        """
        self.dataset_name = dataset_name
        self.config = config
        
        if custom_config:
            self._update_config(custom_config)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.climate_engineer = ClimateFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.interaction_engineer = InteractionFeatureEngineer()
        self.target_creator = PathwayTargetCreator()
        self.model_optimizer = ModelOptimizer()
        self.shap_analyzer = SHAPAnalyzer()
        self.hypothesis_generator = HypothesisGenerator()
        
        # Analysis results
        self.results = {}
        self.models = {}
        self.data = None
        self.features = None
        self.targets = None
    
    def _update_config(self, custom_config: Dict):
        """Update configuration with custom settings."""
        for key, value in custom_config.items():
            if hasattr(self.config.analysis, key):
                setattr(self.config.analysis, key, value)
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load and validate dataset."""
        print("🔄 Loading Data...")
        
        if self.dataset_name:
            dataset_config = self.config.load_dataset_config(self.dataset_name)
            self.data = self.data_loader.load_dataset(dataset_config)
        elif data_path:
            self.data = self.data_loader.load_csv(data_path)
        else:
            raise ValueError("Must specify either dataset_name or data_path")
        
        print(f"✅ Data loaded: {self.data.shape}")
        return self.data
    
    def engineer_features(self, use_temporal: bool = True, 
                         use_interactions: bool = True) -> pd.DataFrame:
        """Engineer comprehensive feature set."""
        print("🔧 Engineering Features...")
        
        features = self.data.copy()
        
        # Climate features
        features = self.climate_engineer.create_features(features)
        
        # Temporal features
        if use_temporal:
            features = self.temporal_engineer.create_features(features)
        
        # Interaction features
        if use_interactions:
            features = self.interaction_engineer.create_features(features)
        
        # Filter based on missing data
        missing_pct = features.isnull().sum() / len(features)
        valid_features = missing_pct[missing_pct <= 0.7].index.tolist()
        features = features[valid_features]
        
        self.features = features
        print(f"✅ Features engineered: {features.shape[1]} features")
        return features
    
    def create_targets(self, pathways: List[str]) -> Dict[str, pd.Series]:
        """Create pathway-specific target variables."""
        print("🎯 Creating Target Variables...")
        
        targets = {}
        for pathway in pathways:
            if pathway in self.config.pathways:
                target = self.target_creator.create_pathway_target(
                    self.features, pathway, self.config.pathways[pathway]
                )
                targets[pathway] = target
                print(f"   {pathway}: {target.describe().iloc[1]:.3f} ± {target.describe().iloc[2]:.3f}")
        
        self.targets = targets
        return targets
    
    def run_pathway_analysis(self, pathway: str, 
                           explain_predictions: bool = True) -> Dict[str, Any]:
        """Run complete analysis for a single pathway."""
        print(f"🔬 Analyzing {pathway.title()} Pathway...")
        
        # Get pathway-specific predictors
        predictors = self.config.get_pathway_predictors(pathway)
        available_predictors = [p for p in predictors if p in self.features.columns]
        
        # Add interaction features
        interaction_features = self.config.get_interaction_features(pathway)
        available_interactions = [i for i in interaction_features if i in self.features.columns]
        all_predictors = available_predictors + available_interactions
        
        # Prepare data
        X = self.features[all_predictors].copy()
        y = self.targets[pathway].copy()
        
        # Remove missing target values
        mask = ~(y.isna() | np.isinf(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"   Clean dataset: {X_clean.shape[0]} participants, {X_clean.shape[1]} features")
        
        # Data preprocessing
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Split data temporally if date column available
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y_clean, 
            test_size=self.config.analysis.test_size, 
            random_state=self.config.analysis.random_seed
        )
        
        # Model optimization
        best_model, best_params, cv_scores = self.model_optimizer.optimize_pathway_model(
            X_train, y_train, pathway
        )
        
        # Model evaluation
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'n_features': X_clean.shape[1],
            'n_samples': X_clean.shape[0]
        }
        
        print(f"   Model Performance: R² = {metrics['test_r2']:.3f}, RMSE = {metrics['test_rmse']:.3f}")
        
        # XAI analysis
        xai_results = {}
        if explain_predictions and metrics['test_r2'] > self.config.analysis.min_predictive_threshold:
            print("   🔍 Running XAI Analysis...")
            xai_results = self.shap_analyzer.analyze_model(
                best_model, X_train, X_test, y_test, pathway
            )
            
            # Generate hypotheses
            hypotheses = self.hypothesis_generator.generate_hypotheses(
                xai_results, pathway, X_train.columns
            )
            xai_results['hypotheses'] = hypotheses
        
        # Store results
        pathway_results = {
            'model': best_model,
            'best_params': best_params,
            'metrics': metrics,
            'feature_names': X_clean.columns.tolist(),
            'predictor_categories': self._categorize_features(X_clean.columns),
            'xai_results': xai_results,
            'data_info': {
                'n_total': len(self.features),
                'n_clean': X_clean.shape[0],
                'missing_target_pct': (mask.sum() / len(mask)) * 100
            }
        }
        
        self.results[pathway] = pathway_results
        self.models[pathway] = best_model
        
        return pathway_results
    
    def run_analysis(self, pathways: List[str] = None, 
                    data_path: Optional[str] = None,
                    use_temporal_features: bool = True,
                    use_interaction_features: bool = True,
                    explain_predictions: bool = True) -> Dict[str, Any]:
        """Run complete heat-health XAI analysis."""
        print("🚀 HEAT-HEALTH XAI ANALYSIS PIPELINE")
        print("=" * 50)
        
        # Default pathways
        if pathways is None:
            pathways = ['inflammatory', 'metabolic', 'cardiovascular']
        
        # Load data
        self.load_data(data_path)
        
        # Engineer features
        self.engineer_features(use_temporal_features, use_interaction_features)
        
        # Create targets
        self.create_targets(pathways)
        
        # Generate descriptive statistics
        desc_stats = self._generate_descriptive_stats()
        
        # Run pathway analyses
        for pathway in pathways:
            self.run_pathway_analysis(pathway, explain_predictions)
        
        # Cross-pathway comparison
        comparison = self._compare_pathways(pathways)
        
        # Compile final results
        final_results = {
            'pathways': self.results,
            'descriptive_stats': desc_stats,
            'comparison': comparison,
            'config': self.config.analysis.__dict__,
            'data_info': {
                'dataset': self.dataset_name,
                'total_samples': len(self.data),
                'total_features': len(self.features.columns),
                'pathways_analyzed': pathways
            }
        }
        
        return final_results
    
    def generate_report(self, results: Dict[str, Any], 
                       output_dir: Optional[Path] = None) -> Path:
        """Generate comprehensive analysis report."""
        if output_dir is None:
            output_dir = self.config.create_output_dir("heat_analysis")
        
        print(f"📊 Generating Report: {output_dir}")
        
        # Save configuration
        self.config.save_config(output_dir)
        
        # Generate pathway reports
        for pathway, pathway_results in results['pathways'].items():
            self._generate_pathway_report(pathway, pathway_results, output_dir)
        
        # Generate comparison report
        self._generate_comparison_report(results['comparison'], output_dir)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        print(f"✅ Reports saved to: {output_dir}")
        return output_dir
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'climate': [],
            'demographic': [],
            'health': [],
            'temporal': [],
            'interactions': []
        }
        
        for feature in feature_names:
            if 'climate_' in feature:
                if '_lag' in feature or '_rolling' in feature:
                    categories['temporal'].append(feature)
                elif '_x_' in feature:
                    categories['interactions'].append(feature)
                else:
                    categories['climate'].append(feature)
            elif 'demographic_' in feature or 'anthropometric_' in feature:
                if '_x_' in feature:
                    categories['interactions'].append(feature)
                else:
                    categories['demographic'].append(feature)
            elif any(h in feature for h in ['metabolic_', 'inflammatory_', 'cardiovascular_', 'renal_']):
                categories['health'].append(feature)
        
        return categories
    
    def _generate_descriptive_stats(self) -> pd.DataFrame:
        """Generate descriptive statistics table."""
        key_vars = {
            'demographic_age': 'Age (years)',
            'demographic_bmi': 'BMI (kg/m²)',
            'climate_mean_temperature': 'Mean Temperature (°C)',
            'climate_max_temperature': 'Max Temperature (°C)',
            'climate_heat_index': 'Heat Index',
            'inflammatory_crp': 'C-Reactive Protein (mg/L)',
            'cardiovascular_systolic_bp_average': 'Systolic BP (mmHg)',
            'metabolic_glucose': 'Glucose (mmol/L)'
        }
        
        desc_stats = []
        for var, label in key_vars.items():
            if var in self.features.columns:
                data = self.features[var].dropna()
                if len(data) > 0:
                    desc_stats.append({
                        'Variable': label,
                        'N': len(data),
                        'Mean': f"{data.mean():.2f}",
                        'SD': f"{data.std():.2f}",
                        'Median': f"{data.median():.2f}",
                        'IQR': f"{data.quantile(0.75) - data.quantile(0.25):.2f}",
                        'Min': f"{data.min():.2f}",
                        'Max': f"{data.max():.2f}",
                        'Missing (%)': f"{(self.features[var].isna().sum() / len(self.features) * 100):.1f}"
                    })
        
        return pd.DataFrame(desc_stats)
    
    def _compare_pathways(self, pathways: List[str]) -> Dict[str, Any]:
        """Compare results across pathways."""
        comparison = {
            'performance': {},
            'top_features': {},
            'common_predictors': [],
            'pathway_specific': {}
        }
        
        # Performance comparison
        for pathway in pathways:
            if pathway in self.results:
                metrics = self.results[pathway]['metrics']
                comparison['performance'][pathway] = {
                    'r2': metrics['test_r2'],
                    'rmse': metrics['test_rmse'],
                    'n_features': metrics['n_features'],
                    'n_samples': metrics['n_samples']
                }
        
        return comparison
    
    def _generate_pathway_report(self, pathway: str, results: Dict[str, Any], 
                               output_dir: Path):
        """Generate individual pathway report."""
        # Implementation for detailed pathway reporting
        pass
    
    def _generate_comparison_report(self, comparison: Dict[str, Any], 
                                  output_dir: Path):
        """Generate cross-pathway comparison report."""
        # Implementation for comparison reporting
        pass
    
    def _generate_summary_report(self, results: Dict[str, Any], 
                               output_dir: Path):
        """Generate executive summary report."""
        # Implementation for summary reporting
        pass