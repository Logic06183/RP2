#!/usr/bin/env python
"""
Hypothesis Generation Module

Generates novel scientific hypotheses from XAI analysis results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

class HypothesisGenerator:
    """Generates scientific hypotheses from XAI results."""
    
    def __init__(self):
        self.hypothesis_templates = {
            'climate_interaction': {
                'template': "Climate variable {climate_var} interacts with {health_var} to influence {pathway} response",
                'evidence_threshold': 0.05  # Top 5% importance
            },
            'temporal_memory': {
                'template': "{lag_period}-day climate memory effects significantly influence {pathway} vulnerability",
                'evidence_threshold': 0.03
            },
            'demographic_modifier': {
                'template': "{demographic_var} modifies the relationship between climate exposure and {pathway} response",
                'evidence_threshold': 0.04
            },
            'threshold_effect': {
                'template': "Heat stress threshold effects are evident in {pathway} pathway at {threshold} levels",
                'evidence_threshold': 0.03
            }
        }
    
    def generate_hypotheses(self, xai_results: Dict[str, Any], 
                           pathway: str,
                           feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Generate scientific hypotheses from XAI analysis.
        
        Args:
            xai_results: Results from SHAP analysis
            pathway: Pathway name
            feature_names: List of feature names
            
        Returns:
            List of hypothesis dictionaries
        """
        if not xai_results or 'feature_importance' not in xai_results:
            return []
        
        hypotheses = []
        feature_importance = xai_results['feature_importance']
        top_features = xai_results.get('top_features', {})
        
        # Generate climate interaction hypotheses
        hypotheses.extend(self._generate_climate_interaction_hypotheses(
            feature_importance, pathway, top_features
        ))
        
        # Generate temporal memory hypotheses
        hypotheses.extend(self._generate_temporal_hypotheses(
            feature_importance, pathway
        ))
        
        # Generate demographic modifier hypotheses
        hypotheses.extend(self._generate_demographic_hypotheses(
            feature_importance, pathway, top_features
        ))
        
        # Generate threshold effect hypotheses
        hypotheses.extend(self._generate_threshold_hypotheses(
            feature_importance, pathway
        ))
        
        # Rank hypotheses by evidence strength
        hypotheses.sort(key=lambda x: x['evidence_strength'], reverse=True)
        
        return hypotheses[:10]  # Return top 10 hypotheses
    
    def _generate_climate_interaction_hypotheses(self, feature_importance: pd.DataFrame,
                                               pathway: str,
                                               top_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses about climate-health interactions."""
        hypotheses = []
        
        # Look for interaction terms in top features
        interaction_features = top_features.get('feature_categories', {}).get('interactions', [])
        
        for interaction in interaction_features:
            if '_x_' in interaction:
                parts = interaction.split('_x_')
                if len(parts) == 2:
                    climate_var = parts[0].replace('climate_', '').replace('_', ' ')
                    health_var = parts[1].replace('demographic_', '').replace('_', ' ')
                    
                    # Get importance score
                    importance_row = feature_importance[feature_importance['feature'] == interaction]
                    if not importance_row.empty:
                        importance_pct = importance_row['importance_pct'].iloc[0]
                        
                        hypotheses.append({
                            'type': 'climate_interaction',
                            'hypothesis': f"Climate {climate_var} interacts with {health_var} to influence {pathway} response",
                            'evidence_strength': importance_pct / 100,
                            'supporting_feature': interaction,
                            'importance_score': importance_pct,
                            'mechanism': f"Individual variation in {health_var} modifies physiological response to {climate_var} exposure"
                        })
        
        return hypotheses
    
    def _generate_temporal_hypotheses(self, feature_importance: pd.DataFrame,
                                    pathway: str) -> List[Dict[str, Any]]:
        """Generate hypotheses about temporal climate memory effects."""
        hypotheses = []
        
        # Look for lag features in top performers
        lag_features = feature_importance[
            feature_importance['feature'].str.contains('_lag\d+d')
        ].head(5)
        
        for _, row in lag_features.iterrows():
            feature = row['feature']
            importance_pct = row['importance_pct']
            
            # Extract lag period
            import re
            lag_match = re.search(r'_lag(\d+)d', feature)
            if lag_match:
                lag_period = lag_match.group(1)
                climate_var = feature.split('_lag')[0].replace('climate_', '').replace('_', ' ')
                
                hypotheses.append({
                    'type': 'temporal_memory',
                    'hypothesis': f"{lag_period}-day {climate_var} memory effects significantly influence {pathway} vulnerability",
                    'evidence_strength': importance_pct / 100,
                    'supporting_feature': feature,
                    'importance_score': importance_pct,
                    'mechanism': f"Physiological adaptation to {climate_var} exposure requires {lag_period} days to manifest in {pathway} biomarkers"
                })
        
        return hypotheses
    
    def _generate_demographic_hypotheses(self, feature_importance: pd.DataFrame,
                                       pathway: str,
                                       top_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypotheses about demographic modifiers."""
        hypotheses = []
        
        # Look for demographic features in top performers
        demographic_features = top_features.get('feature_categories', {}).get('demographic', [])
        
        for demo_feature in demographic_features:
            importance_row = feature_importance[feature_importance['feature'] == demo_feature]
            if not importance_row.empty:
                importance_pct = importance_row['importance_pct'].iloc[0]
                demo_var = demo_feature.replace('demographic_', '').replace('anthropometric_', '').replace('_', ' ')
                
                hypotheses.append({
                    'type': 'demographic_modifier',
                    'hypothesis': f"{demo_var.title()} modifies the relationship between climate exposure and {pathway} response",
                    'evidence_strength': importance_pct / 100,
                    'supporting_feature': demo_feature,
                    'importance_score': importance_pct,
                    'mechanism': f"Individual variation in {demo_var} affects thermoregulatory capacity and {pathway} vulnerability"
                })
        
        return hypotheses
    
    def _generate_threshold_hypotheses(self, feature_importance: pd.DataFrame,
                                     pathway: str) -> List[Dict[str, Any]]:
        """Generate hypotheses about heat stress thresholds."""
        hypotheses = []
        
        # Look for heat stress indicator features
        threshold_features = feature_importance[
            feature_importance['feature'].str.contains('heat_stress|threshold')
        ].head(3)
        
        for _, row in threshold_features.iterrows():
            feature = row['feature']
            importance_pct = row['importance_pct']
            
            if 'moderate' in feature:
                threshold_level = 'moderate (30°C+)'
            elif 'severe' in feature:
                threshold_level = 'severe (35°C+)'
            else:
                threshold_level = 'clinical'
            
            hypotheses.append({
                'type': 'threshold_effect',
                'hypothesis': f"Heat stress threshold effects are evident in {pathway} pathway at {threshold_level} levels",
                'evidence_strength': importance_pct / 100,
                'supporting_feature': feature,
                'importance_score': importance_pct,
                'mechanism': f"Non-linear {pathway} response occurs when heat exposure exceeds {threshold_level} thresholds"
            })
        
        return hypotheses
    
    def format_hypotheses_report(self, hypotheses: List[Dict[str, Any]], 
                               pathway: str) -> str:
        """Format hypotheses into a readable report."""
        
        if not hypotheses:
            return f"No significant hypotheses generated for {pathway} pathway."
        
        report = f"# Novel Scientific hypotheses - {pathway.title()} Pathway\n\n"
        
        for i, hyp in enumerate(hypotheses, 1):
            report += f"## Hypothesis {i}: {hyp['type'].replace('_', ' ').title()}\n\n"
            report += f"**Statement:** {hyp['hypothesis']}\n\n"
            report += f"**Evidence Strength:** {hyp['evidence_strength']:.3f} ({hyp['importance_score']:.1f}% feature importance)\n\n"
            report += f"**Supporting Feature:** `{hyp['supporting_feature']}`\n\n"
            report += f"**Proposed Mechanism:** {hyp['mechanism']}\n\n"
            report += f"**Research Priority:** {'High' if hyp['evidence_strength'] > 0.05 else 'Medium' if hyp['evidence_strength'] > 0.03 else 'Low'}\n\n"
            report += "---\n\n"
        
        return report