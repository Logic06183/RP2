#!/usr/bin/env python
"""
Interaction Feature Engineering Module

Creates climate-health and climate-demographic interaction features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

class InteractionFeatureEngineer:
    """Creates climate-health interaction features."""
    
    def __init__(self):
        self.created_features = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create climate-health interaction features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        
        # Climate × demographic interactions
        df_new = self._create_climate_demographic_interactions(df_new)
        
        # Climate × health interactions  
        df_new = self._create_climate_health_interactions(df_new)
        
        print(f"   Created {len(self.created_features)} interaction features")
        return df_new
    
    def _create_climate_demographic_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create climate × demographic interaction terms."""
        
        # Temperature × BMI interactions
        temp_features = [
            'climate_mean_temperature',
            'climate_max_temperature', 
            'climate_min_temperature'
        ]
        
        demo_features = [
            'demographic_bmi',
            'demographic_age',
            'anthropometric_waist_circumference'
        ]
        
        for temp in temp_features:
            if temp not in df.columns:
                continue
                
            for demo in demo_features:
                if demo not in df.columns:
                    continue
                    
                interaction_name = f"{temp}_x_{demo}"
                df[interaction_name] = df[temp] * df[demo]
                self.created_features.append(interaction_name)
        
        # Heat index × demographic interactions
        heat_features = [
            'climate_heat_index',
            'climate_humidex',
            'climate_wbgt'
        ]
        
        for heat in heat_features:
            if heat not in df.columns:
                continue
                
            if 'demographic_bmi' in df.columns:
                interaction_name = f"{heat}_x_demographic_bmi"
                df[interaction_name] = df[heat] * df['demographic_bmi']
                self.created_features.append(interaction_name)
        
        return df
    
    def _create_climate_health_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create climate × health biomarker interactions."""
        
        climate_features = [
            'climate_mean_temperature',
            'climate_heat_index'
        ]
        
        health_features = [
            'inflammatory_crp',
            'cardiovascular_systolic_bp_average',
            'metabolic_glucose'
        ]
        
        for climate in climate_features:
            if climate not in df.columns:
                continue
                
            for health in health_features:
                if health not in df.columns:
                    continue
                    
                interaction_name = f"{climate}_x_{health}"
                df[interaction_name] = df[climate] * df[health]
                self.created_features.append(interaction_name)
        
        return df