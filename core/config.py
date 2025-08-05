#!/usr/bin/env python
"""
Heat-Health XAI Framework - Configuration Management

Centralized configuration system for multi-dataset heat vulnerability analysis.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    description: str
    file_path: str
    date_column: str
    id_column: Optional[str] = None
    climate_prefix: str = "climate_"
    health_prefix: str = "biomarker_"
    demographic_prefix: str = "demo_"

@dataclass
class PathwayConfig:
    """Configuration for a specific physiological pathway."""
    target: str
    transform: str = "none"  # "none", "log", "sqrt", "standardize"
    exclusions: List[str] = None
    thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.exclusions is None:
            self.exclusions = []
        if self.thresholds is None:
            self.thresholds = {}

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    random_seed: int = 42
    lag_periods: List[int] = None
    rolling_windows: List[int] = None
    interaction_terms: List[str] = None
    model_types: List[str] = None
    cv_folds: int = 5
    test_size: float = 0.3
    shap_sample_size: int = 1000
    min_predictive_threshold: float = 0.01
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [7, 14, 21, 28]
        if self.rolling_windows is None:
            self.rolling_windows = [3, 7, 14]
        if self.interaction_terms is None:
            self.interaction_terms = [
                'climate_temp_x_bmi', 
                'climate_temp_x_age',
                'climate_humidity_x_bmi'
            ]
        if self.model_types is None:
            self.model_types = ['random_forest', 'gradient_boosting', 'elastic_net']

class HeatHealthConfig:
    """Main configuration manager for heat-health analysis framework."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.datasets_dir = self.base_dir / "datasets"
        self.outputs_dir = self.base_dir / "outputs" 
        self.figures_dir = self.base_dir / "figures"
        
        # Default analysis configuration
        self.analysis = AnalysisConfig()
        
        # Default pathway configurations
        self.pathways = {
            'inflammatory': PathwayConfig(
                target='inflammatory_crp',
                transform='log',
                exclusions=[
                    'inflammatory_wbc', 'inflammatory_neutrophils',
                    'inflammatory_lymphocytes', 'inflammatory_il6'
                ],
                thresholds={'high': 5.0, 'moderate': 3.0}
            ),
            'metabolic': PathwayConfig(
                target='metabolic_glucose',
                transform='none',
                exclusions=[
                    'metabolic_hba1c', 'metabolic_insulin',
                    'metabolic_triglycerides_ratio'
                ],
                thresholds={'high': 7.0, 'moderate': 6.1}
            ),
            'cardiovascular': PathwayConfig(
                target='cardiovascular_systolic_bp_average',
                transform='none',
                exclusions=[
                    'cardiovascular_diastolic_bp_average',
                    'cardiovascular_pulse_pressure',
                    'cardiovascular_mean_arterial_pressure'
                ],
                thresholds={'high': 140, 'moderate': 130}
            ),
            'renal': PathwayConfig(
                target='renal_creatinine',
                transform='none',
                exclusions=[
                    'renal_egfr', 'renal_bun', 'renal_uric_acid'
                ],
                thresholds={'high': 120, 'moderate': 100}
            )
        }
        
        # Climate feature groups
        self.climate_features = {
            'temperature': [
                'climate_mean_temperature', 
                'climate_max_temperature', 
                'climate_min_temperature'
            ],
            'humidity': [
                'climate_mean_humidity',
                'climate_relative_humidity'
            ],
            'heat_indices': [
                'climate_heat_index',
                'climate_humidex', 
                'climate_apparent_temp',
                'climate_wbgt'
            ],
            'derived': [
                'climate_temp_range',
                'climate_heat_stress_moderate',
                'climate_heat_stress_severe'
            ]
        }
        
        # Demographic features that are safe across pathways
        self.demographic_features = [
            'demographic_age',
            'demographic_gender',
            'demographic_bmi',
            'anthropometric_weight',
            'anthropometric_height',
            'anthropometric_waist_circumference'
        ]
    
    def load_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Load configuration for a specific dataset."""
        config_path = self.datasets_dir / f"{dataset_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return DatasetConfig(**config_data['dataset'])
    
    def get_pathway_predictors(self, pathway: str, include_climate: bool = True, 
                             include_demographics: bool = True,
                             include_other_health: bool = True) -> List[str]:
        """
        Get safe predictor features for a specific pathway.
        
        Args:
            pathway: Name of physiological pathway
            include_climate: Include climate features
            include_demographics: Include demographic features
            include_other_health: Include health features from other pathways
            
        Returns:
            List of safe predictor feature names
        """
        predictors = []
        
        if include_climate:
            # Add all climate features
            for feature_group in self.climate_features.values():
                predictors.extend(feature_group)
        
        if include_demographics:
            predictors.extend(self.demographic_features)
        
        if include_other_health and pathway in self.pathways:
            # Add health features from other pathways (avoiding target leakage)
            pathway_config = self.pathways[pathway]
            target_feature = pathway_config.target
            exclusions = set(pathway_config.exclusions + [target_feature])
            
            # Add features from all pathways except current one
            for other_pathway, other_config in self.pathways.items():
                if other_pathway != pathway:
                    if other_config.target not in exclusions:
                        predictors.append(other_config.target)
        
        return predictors
    
    def get_interaction_features(self, pathway: str) -> List[str]:
        """Get climate-health interaction features for a pathway."""
        interactions = []
        
        # Temperature × demographic interactions
        temp_features = self.climate_features['temperature']
        demo_features = ['demographic_bmi', 'demographic_age']
        
        for temp in temp_features:
            for demo in demo_features:
                interactions.append(f"{temp}_x_{demo}")
        
        # Heat index × BMI interactions
        heat_features = self.climate_features['heat_indices']
        for heat in heat_features:
            interactions.append(f"{heat}_x_demographic_bmi")
        
        return interactions
    
    def get_temporal_features(self, base_features: List[str]) -> List[str]:
        """Generate temporal lag feature names."""
        temporal_features = []
        
        for feature in base_features:
            if 'climate_' in feature:
                for lag in self.analysis.lag_periods:
                    temporal_features.append(f"{feature}_lag{lag}d")
                
                for window in self.analysis.rolling_windows:
                    temporal_features.append(f"{feature}_rolling{window}d")
        
        return temporal_features
    
    def create_output_dir(self, analysis_name: str) -> Path:
        """Create timestamped output directory."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.outputs_dir / f"{analysis_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_config(self, output_dir: Path):
        """Save current configuration to output directory."""
        config_dict = {
            'analysis': self.analysis.__dict__,
            'pathways': {k: v.__dict__ for k, v in self.pathways.items()},
            'climate_features': self.climate_features,
            'demographic_features': self.demographic_features
        }
        
        config_path = output_dir / "analysis_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Global configuration instance
config = HeatHealthConfig()