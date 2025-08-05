#!/usr/bin/env python
"""
Climate Feature Engineering Module

Creates climate-based features for heat vulnerability analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ClimateFeatureEngineer:
    """Creates climate-derived features."""
    
    def __init__(self):
        self.created_features = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive climate feature set.
        
        Args:
            df: Input DataFrame with climate variables
            
        Returns:
            DataFrame with added climate features
        """
        df_new = df.copy()
        
        # Temperature range features
        df_new = self._create_temperature_ranges(df_new)
        
        # Heat stress indicators
        df_new = self._create_heat_stress_indicators(df_new)
        
        # Thermal comfort indices
        df_new = self._create_thermal_comfort_indices(df_new)
        
        print(f"   Created {len(self.created_features)} climate features")
        return df_new
    
    def _create_temperature_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temperature range and variability features."""
        temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
        
        if 'climate_max_temperature' in df.columns and 'climate_min_temperature' in df.columns:
            df['climate_temp_range'] = df['climate_max_temperature'] - df['climate_min_temperature']
            self.created_features.append('climate_temp_range')
        
        return df
    
    def _create_heat_stress_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary and continuous heat stress indicators."""
        if 'climate_max_temperature' in df.columns:
            # Heat stress thresholds
            df['climate_heat_stress_moderate'] = (df['climate_max_temperature'] > 30).astype(int)
            df['climate_heat_stress_severe'] = (df['climate_max_temperature'] > 35).astype(int)
            
            self.created_features.extend([
                'climate_heat_stress_moderate',
                'climate_heat_stress_severe'
            ])
        
        return df
    
    def _create_thermal_comfort_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional thermal comfort indices if base data available."""
        # Placeholder for additional indices
        # Could include: Effective Temperature, Universal Thermal Climate Index (UTCI), etc.
        return df