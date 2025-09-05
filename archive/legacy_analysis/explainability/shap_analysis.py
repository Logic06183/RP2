#!/usr/bin/env python
"""
SHAP Analysis Module

Explainable AI analysis using SHAP values for heat vulnerability models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    """Performs SHAP analysis for model interpretability."""
    
    def __init__(self):
        self.shap_values = {}
        self.explainers = {}
    
    def analyze_model(self, model: Any, 
                     X_train: pd.DataFrame,
                     X_test: pd.DataFrame, 
                     y_test: pd.Series,
                     pathway: str,
                     sample_size: int = 1000) -> Dict[str, Any]:
        """
        Perform comprehensive SHAP analysis.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_test: Test targets
            pathway: Pathway name
            sample_size: Sample size for SHAP analysis
            
        Returns:
            Dictionary with SHAP analysis results
        """
        print(f"   🔍 Running SHAP analysis for {pathway}...")
        
        # Limit sample size for efficiency
        if len(X_test) > sample_size:
            sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test.iloc[sample_idx]
            y_sample = y_test.iloc[sample_idx]
        else:
            X_sample = X_test
            y_sample = y_test
        
        try:
            # Create SHAP explainer
            explainer = self._create_explainer(model, X_train)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Store results
            self.explainers[pathway] = explainer
            self.shap_values[pathway] = shap_values
            
            # Analyze results
            results = {
                'shap_values': shap_values,
                'feature_importance': self._calculate_feature_importance(shap_values, X_sample.columns),
                'top_features': self._get_top_features(shap_values, X_sample.columns, top_k=10),
                'feature_interactions': self._analyze_feature_interactions(shap_values, X_sample.columns),
                'sample_data': X_sample,
                'sample_predictions': model.predict(X_sample),
                'sample_targets': y_sample
            }
            
            print(f"   ✅ SHAP analysis complete ({len(X_sample)} samples)")
            return results
            
        except Exception as e:
            print(f"   ❌ SHAP analysis failed: {str(e)}")
            return {}
    
    def _create_explainer(self, model: Any, X_train: pd.DataFrame) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model type."""
        
        model_name = type(model).__name__.lower()
        
        if 'forest' in model_name or 'tree' in model_name or 'boost' in model_name:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
        else:
            # Linear or other models
            explainer = shap.Explainer(model, X_train.sample(min(100, len(X_train))))
        
        return explainer
    
    def _calculate_feature_importance(self, shap_values: np.ndarray, 
                                    feature_names: List[str]) -> pd.DataFrame:
        """Calculate mean absolute SHAP values as feature importance."""
        
        if len(shap_values.shape) == 2:
            # Single output
            importance = np.abs(shap_values).mean(axis=0)
        else:
            # Multiple outputs (shouldn't happen for regression)
            importance = np.abs(shap_values).mean(axis=(0, 2))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Normalize to percentages
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df
    
    def _get_top_features(self, shap_values: np.ndarray, 
                         feature_names: List[str], 
                         top_k: int = 10) -> Dict[str, Any]:
        """Get top contributing features."""
        
        importance_df = self._calculate_feature_importance(shap_values, feature_names)
        top_features = importance_df.head(top_k)
        
        # Categorize features
        categories = {
            'climate': [],
            'demographic': [],
            'health': [],
            'temporal': [],
            'interactions': []
        }
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            if 'climate_' in feature:
                if '_lag' in feature or '_rolling' in feature:
                    categories['temporal'].append(feature)
                elif '_x_' in feature:
                    categories['interactions'].append(feature)
                else:
                    categories['climate'].append(feature)
            elif 'demographic_' in feature or 'anthropometric_' in feature:
                categories['demographic'].append(feature)
            elif any(h in feature for h in ['metabolic_', 'inflammatory_', 'cardiovascular_']):
                categories['health'].append(feature)
        
        return {
            'top_features_df': top_features,
            'feature_categories': categories,
            'climate_dominance': len(categories['climate']) / top_k,
            'temporal_importance': len(categories['temporal']) / top_k,
            'interaction_importance': len(categories['interactions']) / top_k
        }
    
    def _analyze_feature_interactions(self, shap_values: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature interactions from SHAP values."""
        
        # Simple interaction analysis based on SHAP value correlations
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        correlation_matrix = shap_df.corr()
        
        # Find strong correlations (potential interactions)
        strong_correlations = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.3:  # Threshold for "strong" correlation
                    strong_correlations.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'correlation': corr
                    })
        
        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_interactions': strong_correlations[:10],  # Top 10
            'n_strong_interactions': len(strong_correlations)
        }
    
    def create_summary_plot(self, pathway: str, output_dir: Path) -> Path:
        """Create SHAP summary plot."""
        
        if pathway not in self.shap_values:
            raise ValueError(f"No SHAP values available for pathway: {pathway}")
        
        shap_values = self.shap_values[pathway]
        # Get corresponding sample data from results
        # This would need to be stored during analysis
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, show=False)
        plt.title(f'SHAP Feature Importance - {pathway.title()} Pathway')
        plt.tight_layout()
        
        plot_path = output_dir / f'shap_summary_{pathway}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path