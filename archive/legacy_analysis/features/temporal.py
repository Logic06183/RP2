#!/usr/bin/env python
"""
Temporal Feature Engineering Module

Creates lag and rolling window features for climate variables.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class TemporalFeatureEngineer:
    """Creates temporal lag and rolling window features."""
    
    def __init__(self):
        self.created_features = []
    
    def create_features(self, df: pd.DataFrame, 
                       date_column: Optional[str] = None,
                       lag_periods: List[int] = None,
                       rolling_windows: List[int] = None) -> pd.DataFrame:
        """
        Create temporal features for climate variables.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column for temporal ordering
            lag_periods: List of lag periods in days
            rolling_windows: List of rolling window sizes in days
            
        Returns:
            DataFrame with temporal features
        """
        if lag_periods is None:
            lag_periods = [7, 14, 21, 28]
        if rolling_windows is None:
            rolling_windows = [3, 7, 14]
        
        df_new = df.copy()
        
        # Identify climate features
        climate_features = [col for col in df.columns if 'climate_' in col]
        
        if not climate_features:
            print("   No climate features found for temporal processing")
            return df_new
        
        # Create lag features
        df_new = self._create_lag_features(df_new, climate_features, lag_periods, date_column)
        
        # Create rolling window features
        df_new = self._create_rolling_features(df_new, climate_features, rolling_windows, date_column)
        
        print(f"   Created {len(self.created_features)} temporal features")
        return df_new
    
    def _create_lag_features(self, df: pd.DataFrame, 
                           climate_features: List[str],
                           lag_periods: List[int],
                           date_column: Optional[str]) -> pd.DataFrame:
        """Create lagged climate features."""
        
        for feature in climate_features:
            if feature not in df.columns:
                continue
                
            for lag in lag_periods:
                lag_name = f"{feature}_lag{lag}d"
                
                if date_column and date_column in df.columns:
                    # Proper temporal lag using date column
                    df_sorted = df.sort_values(date_column)
                    df[lag_name] = df_sorted[feature].shift(lag)
                else:
                    # Simple row-based lag (less accurate)
                    df[lag_name] = df[feature].shift(lag)
                
                self.created_features.append(lag_name)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame,
                               climate_features: List[str],
                               rolling_windows: List[int],
                               date_column: Optional[str]) -> pd.DataFrame:
        """Create rolling window averages."""
        
        for feature in climate_features:
            if feature not in df.columns:
                continue
                
            for window in rolling_windows:
                rolling_name = f"{feature}_rolling{window}d"
                
                # Create rolling average
                df[rolling_name] = df[feature].rolling(window=window, min_periods=1).mean()
                self.created_features.append(rolling_name)
        
        return df