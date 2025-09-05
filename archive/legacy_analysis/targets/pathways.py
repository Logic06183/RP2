#!/usr/bin/env python
"""
Pathway Target Creation Module

Creates pathway-specific target variables for heat vulnerability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ..core.config import PathwayConfig

class PathwayTargetCreator:
    """Creates pathway-specific target variables."""
    
    def __init__(self):
        self.created_targets = {}
    
    def create_pathway_target(self, df: pd.DataFrame, 
                            pathway: str,
                            config: PathwayConfig) -> pd.Series:
        """
        Create target variable for specific pathway.
        
        Args:
            df: Input DataFrame
            pathway: Pathway name
            config: Pathway configuration
            
        Returns:
            Target variable Series
        """
        if config.target not in df.columns:
            raise ValueError(f"Target column '{config.target}' not found in data")
        
        target = df[config.target].copy()
        
        # Apply transformation
        if config.transform == 'log':
            target = np.log1p(target.fillna(target.median()))
        elif config.transform == 'sqrt':
            target = np.sqrt(target.fillna(target.median()))
        elif config.transform == 'standardize':
            target = (target - target.mean()) / target.std()
        
        # Store target info
        self.created_targets[pathway] = {
            'original_column': config.target,
            'transform': config.transform,
            'n_valid': target.notna().sum(),
            'missing_pct': (target.isna().sum() / len(target)) * 100
        }
        
        return target