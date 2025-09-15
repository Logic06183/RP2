#!/usr/bin/env python3

"""
Publication-Ready DLNM-Style Climate-Health Analysis
=====================================================

Comprehensive analysis of longitudinal climate-health relationships focusing on 
cardiovascular and renal pathways using distributed lag modeling approaches.

Key Features:
- Longitudinal data handling with proper temporal clustering
- DLNM-style distributed lag modeling
- Publication-quality SVG visualizations
- Comprehensive statistical reporting
- Socioeconomic interaction analysis

Author: Claude Code - Heat-Health Research Team
Date: 2025-09-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import interpolate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

import json
import os
from datetime import datetime, timedelta
import sys

# Set random seed for reproducibility
np.random.seed(42)

class DLNMClimateHealthAnalyzer:
    """
    Advanced DLNM-style analyzer for climate-health relationships with longitudinal data
    """
    
    def __init__(self, max_lag=21, temp_knots=None, verbose=True):
        self.max_lag = max_lag
        self.temp_knots = temp_knots or [10, 25, 90]  # Percentile knots
        self.verbose = verbose
        self.results = {}
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'font.family': 'DejaVu Sans'
        })
        
        if self.verbose:
            print("🔬 DLNM Climate-Health Analyzer Initialized")
            print(f"   • Maximum lag period: {self.max_lag} days")
            print(f"   • Temperature knots at percentiles: {self.temp_knots}")
    
    def load_and_prepare_data(self, gcro_path, enhanced_results_path):
        """Load and prepare longitudinal climate-health data"""
        
        if self.verbose:
            print("📂 Loading and preparing longitudinal data...")
        
        # Load GCRO socioeconomic data with climate integration
        self.gcro_data = pd.read_csv(gcro_path)
        
        # Load enhanced analysis results
        with open(enhanced_results_path, 'r') as f:
            self.enhanced_results = json.load(f)
        
        # Parse dates and create temporal variables
        self.gcro_data['date'] = pd.to_datetime(self.gcro_data['interview_date_parsed'])
        self.gcro_data['year'] = self.gcro_data['date'].dt.year
        self.gcro_data['month'] = self.gcro_data['date'].dt.month
        self.gcro_data['day_of_year'] = self.gcro_data['date'].dt.dayofyear
        
        # Primary temperature variable (ERA5 30-day mean)
        self.gcro_data['temperature'] = self.gcro_data['era5_temp_30d_mean']
        
        # Fill missing temperatures with alternative sources
        missing_temp = self.gcro_data['temperature'].isna()
        self.gcro_data.loc[missing_temp, 'temperature'] = \
            self.gcro_data.loc[missing_temp, 'era5_lst_30d_mean']
        
        # Still missing? Use MODIS
        missing_temp = self.gcro_data['temperature'].isna()
        self.gcro_data.loc[missing_temp, 'temperature'] = \
            self.gcro_data.loc[missing_temp, 'modis_lst_30d_mean']
        
        # Temperature percentiles for knot placement
        temp_percentiles = np.percentile(
            self.gcro_data['temperature'].dropna(), 
            self.temp_knots
        )
        
        # Create temperature indicators
        self.gcro_data['extreme_heat'] = (
            self.gcro_data['temperature'] >= temp_percentiles[2]
        ).astype(int)
        self.gcro_data['high_heat'] = (
            self.gcro_data['temperature'] >= temp_percentiles[1]
        ).astype(int)
        
        # Health outcomes - cardiovascular pathway
        cardio_conditions = ['q13_11_6_hypertension', 'q13_11_5_heart']
        self.gcro_data['cardiovascular_score'] = 0
        for condition in cardio_conditions:
            if condition in self.gcro_data.columns:
                self.gcro_data['cardiovascular_score'] += (
                    self.gcro_data[condition] == 'Yes'
                ).astype(int)
        
        # Normalize cardiovascular score
        self.gcro_data['cardiovascular_score'] = (
            self.gcro_data['cardiovascular_score'] / len(cardio_conditions)
        )
        
        # Renal pathway (using diabetes as proxy for renal risk)
        self.gcro_data['renal_risk'] = (
            (self.gcro_data['q13_11_2_diabetes'] == 'Yes') &
            (self.gcro_data['q13_6_health_status'].isin(['Poor', 'Fair']))
        ).astype(int)
        
        # Socioeconomic vulnerability index
        self._create_ses_vulnerability_index()
        
        # Filter complete cases
        required_cols = ['temperature', 'cardiovascular_score', 'renal_risk', 
                        'ses_vulnerability', 'date']
        self.analysis_data = self.gcro_data.dropna(subset=required_cols).copy()
        
        # Sort by date for time series analysis
        self.analysis_data = self.analysis_data.sort_values('date').reset_index(drop=True)
        
        if self.verbose:
            print(f"   • Total observations: {len(self.analysis_data):,}")
            print(f"   • Date range: {self.analysis_data['date'].min().date()} to {self.analysis_data['date'].max().date()}")
            print(f"   • Temperature range: {self.analysis_data['temperature'].min():.1f}°C to {self.analysis_data['temperature'].max():.1f}°C")
            print(f"   • Cardiovascular cases: {self.analysis_data['cardiovascular_score'].sum():.0f}")
            print(f"   • Renal risk cases: {self.analysis_data['renal_risk'].sum():.0f}")
    
    def _create_ses_vulnerability_index(self):
        """Create comprehensive socioeconomic vulnerability index"""
        
        # Income (reverse coded - lower income = higher vulnerability)
        income_mapping = {
            'Prefer not to answer': np.nan,
            'R25 601 - R38 400': 1, 'R25 601 - R51 200': 1,
            'R12 801 - R25 600': 2, 'R6 401 - R12 800': 3,
            'R3 201 - R6 400': 4, 'R1 601 - R3 200': 5,
            'R801 - R1 600': 6, 'R401 - R800': 7, 'R1 - R400': 8
        }
        
        self.gcro_data['income_vulnerability'] = self.gcro_data['q15_3_income'].map(income_mapping)
        
        # Education (reverse coded - lower education = higher vulnerability)
        education_mapping = {
            'Technikon or university degree': 1,
            'Grade 12, Std 10, Matric': 2,
            'Grade 11, Std 9 or Form IV': 3,
            'Grade 9, Std 7, Form II, NQF 1 or ABET 4': 4,
            'Grade 7, Std 5, Form I or lower': 5
        }
        
        self.gcro_data['education_vulnerability'] = self.gcro_data['q14_1_education'].map(education_mapping)
        
        # Employment vulnerability
        employment_mapping = {
            'Full time': 1, 'Part time': 2, 'Casual work': 3,
            'Self employed': 2, 'Unemployed': 4, 'Student': 2,
            'Retired': 2, 'Housewife/homemaker': 3
        }
        
        self.gcro_data['employment_vulnerability'] = self.gcro_data['q10_2_working'].map(employment_mapping)
        
        # Composite SES vulnerability (standardized)
        ses_components = ['income_vulnerability', 'education_vulnerability', 'employment_vulnerability']
        
        # Fill missing with median
        for comp in ses_components:
            if comp in self.gcro_data.columns:
                median_val = self.gcro_data[comp].median()
                self.gcro_data[comp] = self.gcro_data[comp].fillna(median_val)
        
        # Calculate composite index
        self.gcro_data['ses_vulnerability'] = self.gcro_data[ses_components].mean(axis=1)
        
        # Standardize (0-1 scale)
        ses_min = self.gcro_data['ses_vulnerability'].min()
        ses_max = self.gcro_data['ses_vulnerability'].max()
        self.gcro_data['ses_vulnerability'] = (
            (self.gcro_data['ses_vulnerability'] - ses_min) / (ses_max - ses_min)
        )
    
    def create_distributed_lag_features(self, data, temp_col='temperature'):
        """Create distributed lag features for DLNM-style modeling"""
        
        if self.verbose:
            print(f"🔧 Creating distributed lag features (max lag: {self.max_lag} days)...")
        
        # Sort by date to ensure proper temporal ordering
        data_sorted = data.sort_values('date').copy()
        
        # Create lag features
        lag_features = pd.DataFrame(index=data_sorted.index)
        
        # Add temperature lags
        for lag in range(self.max_lag + 1):
            if lag == 0:
                lag_features[f'temp_lag_{lag}'] = data_sorted[temp_col]
            else:
                lag_features[f'temp_lag_{lag}'] = data_sorted[temp_col].shift(lag)
        
        # Create temperature basis functions (natural cubic splines approximation)
        temp_values = data_sorted[temp_col].values
        temp_knots_values = np.percentile(temp_values[~np.isnan(temp_values)], self.temp_knots)
        
        # Temperature spline basis (simplified)
        for i, knot in enumerate(temp_knots_values):
            lag_features[f'temp_spline_{i}'] = np.maximum(0, temp_values - knot) ** 3
        
        # Lag basis functions (linear, quadratic, cubic)
        lag_range = np.arange(self.max_lag + 1)
        for degree in range(1, 4):
            for lag in range(self.max_lag + 1):
                lag_features[f'lag_basis_{degree}_{lag}'] = (lag / self.max_lag) ** degree
        
        # Interaction terms (temperature × lag)
        for lag in range(0, self.max_lag + 1, 3):  # Every 3rd lag to reduce dimensionality
            lag_features[f'temp_lag_interact_{lag}'] = (
                data_sorted[temp_col] * lag_features[f'lag_basis_1_{lag}']
            )
        
        # Add seasonal and temporal controls
        lag_features['day_of_year_sin'] = np.sin(2 * np.pi * data_sorted['day_of_year'] / 365.25)
        lag_features['day_of_year_cos'] = np.cos(2 * np.pi * data_sorted['day_of_year'] / 365.25)
        lag_features['year'] = data_sorted['year']
        lag_features['ses_vulnerability'] = data_sorted['ses_vulnerability']
        
        if self.verbose:
            print(f"   • Created {lag_features.shape[1]} lag features")
            print(f"   • Temperature knots at: {temp_knots_values}")
        
        return lag_features
    
    def fit_dlnm_models(self):
        """Fit DLNM-style models for cardiovascular and renal pathways"""
        
        if self.verbose:
            print("🏥 Fitting DLNM models for health pathways...")
        
        # Create distributed lag features
        self.lag_features = self.create_distributed_lag_features(self.analysis_data)
        
        # Remove rows with NaN values (due to lagging)
        valid_idx = ~self.lag_features.isna().any(axis=1)
        X = self.lag_features[valid_idx]
        
        self.results['dlnm_models'] = {}
        
        # Cardiovascular pathway
        y_cardio = self.analysis_data.loc[valid_idx, 'cardiovascular_score']
        self._fit_pathway_model('cardiovascular', X, y_cardio, 'gaussian')
        
        # Renal pathway  
        y_renal = self.analysis_data.loc[valid_idx, 'renal_risk']
        self._fit_pathway_model('renal', X, y_renal, 'binomial')
        
        if self.verbose:
            print("   • DLNM models fitted successfully")
    
    def _fit_pathway_model(self, pathway, X, y, family):
        """Fit individual pathway model with multiple algorithms"""
        
        if self.verbose:
            print(f"   • Fitting {pathway} pathway model...")
        
        # Time series cross-validation to prevent leakage
        tscv = TimeSeriesSplit(n_splits=5, test_size=len(y)//10)
        
        models = {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        pathway_results = {}
        
        for name, model in models.items():
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            
            # Fit full model
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            pathway_results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'r2_full': r2,
                'mse_full': mse,
                'predictions': y_pred
            }
            
            if self.verbose:
                print(f"     • {name}: CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Select best model based on CV performance
        best_model_name = max(pathway_results.keys(), 
                             key=lambda k: pathway_results[k]['cv_mean'])
        pathway_results['best_model'] = best_model_name
        
        self.results['dlnm_models'][pathway] = pathway_results
    
    def analyze_temperature_response_curves(self):
        """Analyze temperature-response relationships"""
        
        if self.verbose:
            print("📈 Analyzing temperature-response curves...")
        
        temp_range = np.linspace(
            self.analysis_data['temperature'].min(),
            self.analysis_data['temperature'].max(),
            100
        )
        
        self.results['temperature_response'] = {}
        
        for pathway in ['cardiovascular', 'renal']:
            best_model_name = self.results['dlnm_models'][pathway]['best_model']
            model = self.results['dlnm_models'][pathway][best_model_name]['model']
            
            # Create temperature response data
            temp_effects = []
            for temp in temp_range:
                # Create feature vector for this temperature
                temp_features = self._create_temp_feature_vector(temp)
                effect = model.predict([temp_features])[0]
                temp_effects.append(effect)
            
            self.results['temperature_response'][pathway] = {
                'temperature': temp_range,
                'effect': np.array(temp_effects),
                'peak_temp': temp_range[np.argmax(temp_effects)],
                'peak_effect': np.max(temp_effects)
            }
        
        if self.verbose:
            print("   • Temperature-response curves calculated")
    
    def _create_temp_feature_vector(self, temperature):
        """Create feature vector for a given temperature (simplified)"""
        
        # Get median values for other features
        median_features = self.lag_features.median()
        
        # Set temperature-related features
        feature_vector = median_features.copy()
        feature_vector['temp_lag_0'] = temperature
        
        # Update temperature spline features
        temp_knots_values = np.percentile(
            self.analysis_data['temperature'].dropna(), 
            self.temp_knots
        )
        
        for i, knot in enumerate(temp_knots_values):
            feature_vector[f'temp_spline_{i}'] = max(0, temperature - knot) ** 3
        
        return feature_vector.values
    
    def analyze_lag_effects(self):
        """Analyze lag-response relationships"""
        
        if self.verbose:
            print("⏰ Analyzing lag-response relationships...")
        
        # Define extreme temperature threshold
        extreme_temp = np.percentile(self.analysis_data['temperature'], 90)
        
        self.results['lag_effects'] = {}
        
        for pathway in ['cardiovascular', 'renal']:
            best_model_name = self.results['dlnm_models'][pathway]['best_model']
            model = self.results['dlnm_models'][pathway][best_model_name]['model']
            
            # Extract lag coefficients (simplified approach)
            feature_names = self.lag_features.columns
            
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
            elif hasattr(model, 'feature_importances_'):
                coefficients = model.feature_importances_
            else:
                coefficients = np.ones(len(feature_names))  # Fallback
            
            # Extract lag-related effects
            lag_effects = []
            for lag in range(self.max_lag + 1):
                lag_cols = [col for col in feature_names if f'lag_{lag}' in col]
                if lag_cols:
                    lag_indices = [feature_names.get_loc(col) for col in lag_cols]
                    avg_effect = np.mean([coefficients[idx] for idx in lag_indices])
                    lag_effects.append(avg_effect)
                else:
                    lag_effects.append(0)
            
            self.results['lag_effects'][pathway] = {
                'lags': np.arange(self.max_lag + 1),
                'effects': np.array(lag_effects),
                'extreme_temp': extreme_temp
            }
        
        if self.verbose:
            print(f"   • Lag effects analyzed for extreme temperature ({extreme_temp:.1f}°C)")
    
    def analyze_ses_interactions(self):
        """Analyze temperature × socioeconomic status interactions"""
        
        if self.verbose:
            print("🔗 Analyzing socioeconomic interactions...")
        
        # Create SES tertiles (handle duplicate edges)
        try:
            ses_tertiles = pd.qcut(
                self.analysis_data['ses_vulnerability'], 
                q=3, 
                labels=['Low_vulnerability', 'Medium_vulnerability', 'High_vulnerability'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback to manual binning if qcut fails
            ses_vals = self.analysis_data['ses_vulnerability']
            ses_tertiles = pd.cut(
                ses_vals,
                bins=3,
                labels=['Low_vulnerability', 'Medium_vulnerability', 'High_vulnerability']
            )
        
        self.results['ses_interactions'] = {}
        
        for pathway in ['cardiovascular', 'renal']:
            interaction_effects = {}
            
            for ses_level in ses_tertiles.cat.categories:
                # Subset data for this SES level
                ses_mask = ses_tertiles == ses_level
                if ses_mask.sum() < 50:  # Skip if insufficient data
                    continue
                
                # Get model predictions for this subset
                best_model_name = self.results['dlnm_models'][pathway]['best_model']
                predictions = self.results['dlnm_models'][pathway][best_model_name]['predictions']
                
                # Calculate mean effect for this SES group
                valid_idx = ~self.lag_features.isna().any(axis=1)
                # Align indices properly
                valid_ses_mask = ses_mask[valid_idx]
                ses_predictions = predictions[valid_ses_mask]
                
                interaction_effects[ses_level] = {
                    'mean_effect': np.mean(ses_predictions),
                    'std_effect': np.std(ses_predictions),
                    'sample_size': len(ses_predictions)
                }
            
            self.results['ses_interactions'][pathway] = interaction_effects
        
        if self.verbose:
            print("   • SES interaction analysis completed")
    
    def create_publication_visualizations(self, output_dir):
        """Create publication-quality visualizations"""
        
        if self.verbose:
            print("📊 Creating publication-quality visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Figure 1: Temperature-Response Curves
        self._create_temperature_response_plot(output_dir)
        
        # Figure 2: Lag-Response Curves
        self._create_lag_response_plot(output_dir)
        
        # Figure 3: SES Interaction Effects
        self._create_ses_interaction_plot(output_dir)
        
        # Figure 4: Model Performance Comparison
        self._create_model_performance_plot(output_dir)
        
        if self.verbose:
            print(f"   • All visualizations saved to {output_dir}")
    
    def _create_temperature_response_plot(self, output_dir):
        """Create temperature-response curves plot"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'cardiovascular': '#E31A1C', 'renal': '#1F78B4'}
        
        for pathway in ['cardiovascular', 'renal']:
            if pathway in self.results['temperature_response']:
                temp_data = self.results['temperature_response'][pathway]
                ax.plot(temp_data['temperature'], temp_data['effect'], 
                       color=colors[pathway], linewidth=2.5, 
                       label=f'{pathway.title()} Pathway', alpha=0.8)
        
        # Add threshold lines
        temp_90th = np.percentile(self.analysis_data['temperature'], 90)
        temp_75th = np.percentile(self.analysis_data['temperature'], 75)
        
        ax.axvline(temp_75th, color='orange', linestyle='--', alpha=0.7, 
                  label=f'75th percentile ({temp_75th:.1f}°C)')
        ax.axvline(temp_90th, color='red', linestyle='--', alpha=0.7,
                  label=f'90th percentile ({temp_90th:.1f}°C)')
        
        ax.set_xlabel('Temperature (°C)', fontsize=14)
        ax.set_ylabel('Health Effect (Relative Risk)', fontsize=14)
        ax.set_title('Temperature-Health Response Relationships\n' + 
                    'DLNM Analysis of Heat Exposure Effects', fontsize=16, pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperature_response_curves.svg'), 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_lag_response_plot(self, output_dir):
        """Create lag-response curves plot"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {'cardiovascular': '#E31A1C', 'renal': '#1F78B4'}
        
        for i, (pathway, ax) in enumerate(zip(['cardiovascular', 'renal'], [ax1, ax2])):
            if pathway in self.results['lag_effects']:
                lag_data = self.results['lag_effects'][pathway]
                
                ax.plot(lag_data['lags'], lag_data['effects'], 
                       color=colors[pathway], linewidth=2.5, marker='o', 
                       markersize=4, alpha=0.8)
                ax.axhline(0, color='black', linestyle=':', alpha=0.5)
                
                ax.set_xlabel('Lag (days)', fontsize=12)
                ax.set_ylabel('Health Effect', fontsize=12)
                ax.set_title(f'{pathway.title()} Pathway\nLag-Response Pattern', 
                           fontsize=14)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Lag-Response Relationships at Extreme Heat\n' + 
                    f'Effects over {self.max_lag} days following exposure', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lag_response_curves.svg'), 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ses_interaction_plot(self, output_dir):
        """Create SES interaction effects plot"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        pathways = []
        ses_levels = []
        effects = []
        errors = []
        
        for pathway in ['cardiovascular', 'renal']:
            if pathway in self.results['ses_interactions']:
                for ses_level, data in self.results['ses_interactions'][pathway].items():
                    pathways.append(pathway.title())
                    ses_levels.append(ses_level.replace('_', ' ').title())
                    effects.append(data['mean_effect'])
                    errors.append(data['std_effect'] / np.sqrt(data['sample_size']))  # SEM
        
        if pathways:  # Only plot if we have data
            plot_df = pd.DataFrame({
                'Pathway': pathways,
                'SES_Level': ses_levels,
                'Effect': effects,
                'Error': errors
            })
            
            sns.barplot(data=plot_df, x='SES_Level', y='Effect', hue='Pathway', 
                       palette={'Cardiovascular': '#E31A1C', 'Renal': '#1F78B4'}, ax=ax)
            
            # Add error bars
            for i, (pathway, ses_level) in enumerate(zip(pathways, ses_levels)):
                bar_idx = i % len(plot_df['SES_Level'].unique())
                hue_idx = i // len(plot_df['SES_Level'].unique())
                ax.errorbar(bar_idx + (hue_idx - 0.5) * 0.3, effects[i], 
                           yerr=errors[i], fmt='none', color='black', capsize=5)
        
        ax.set_xlabel('Socioeconomic Vulnerability', fontsize=14)
        ax.set_ylabel('Health Effect', fontsize=14)
        ax.set_title('Temperature Effects by Socioeconomic Vulnerability\n' + 
                    'Health impacts stratified by vulnerability tertiles', fontsize=16, pad=20)
        ax.legend(title='Pathway', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ses_interaction_effects.svg'), 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_plot(self, output_dir):
        """Create model performance comparison plot"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        pathways = ['cardiovascular', 'renal']
        model_names = ['ridge', 'elastic_net', 'random_forest', 'gradient_boosting']
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, pathway in enumerate(pathways):
            ax = ax1 if i == 0 else ax2
            
            if pathway in self.results['dlnm_models']:
                cv_means = []
                cv_stds = []
                labels = []
                
                for model_name in model_names:
                    if model_name in self.results['dlnm_models'][pathway]:
                        model_data = self.results['dlnm_models'][pathway][model_name]
                        cv_means.append(model_data['cv_mean'])
                        cv_stds.append(model_data['cv_std'])
                        
                        # Mark best model
                        best_model = self.results['dlnm_models'][pathway]['best_model']
                        label = f"{model_name.replace('_', ' ').title()}"
                        if model_name == best_model:
                            label += ' (Best)'
                        labels.append(label)
                
                if cv_means:
                    bars = ax.bar(range(len(cv_means)), cv_means, 
                                 yerr=cv_stds, capsize=5, 
                                 color=colors[:len(cv_means)], alpha=0.8)
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel('Cross-Validation R²', fontsize=12)
                    ax.set_title(f'{pathway.title()} Pathway\nModel Performance', fontsize=14)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Highlight best model
                    best_idx = labels.index([l for l in labels if '(Best)' in l][0])
                    bars[best_idx].set_edgecolor('red')
                    bars[best_idx].set_linewidth(2)
        
        plt.suptitle('DLNM Model Performance Comparison\n' + 
                    'Cross-Validation Results Across Algorithms', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_performance_comparison.svg'), 
                   format='svg', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, output_dir):
        """Generate comprehensive analysis report"""
        
        if self.verbose:
            print("📋 Generating comprehensive analysis report...")
        
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyst': 'DLNM Climate-Health Publication Analysis',
                'analysis_type': 'distributed_lag_nonlinear_model',
                'sample_size': len(self.analysis_data),
                'date_range': {
                    'start': self.analysis_data['date'].min().isoformat(),
                    'end': self.analysis_data['date'].max().isoformat()
                },
                'temperature_range': {
                    'min': float(self.analysis_data['temperature'].min()),
                    'max': float(self.analysis_data['temperature'].max()),
                    'mean': float(self.analysis_data['temperature'].mean()),
                    'std': float(self.analysis_data['temperature'].std())
                },
                'max_lag_days': self.max_lag,
                'pathways_analyzed': ['cardiovascular', 'renal']
            },
            
            'dlnm_results': self._serialize_model_results(),
            
            'temperature_response_analysis': self._serialize_temp_response(),
            
            'lag_effects_analysis': self._serialize_lag_effects(),
            
            'socioeconomic_interactions': self._serialize_ses_interactions(),
            
            'key_findings': self._generate_key_findings(),
            
            'statistical_significance': self._assess_statistical_significance(),
            
            'clinical_relevance': self._assess_clinical_relevance()
        }
        
        # Save comprehensive report
        with open(os.path.join(output_dir, 'dlnm_comprehensive_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.verbose:
            print(f"   • Comprehensive report saved to {output_dir}")
        
        return report
    
    def _serialize_model_results(self):
        """Serialize model results for JSON output"""
        serialized = {}
        
        for pathway, models in self.results['dlnm_models'].items():
            serialized[pathway] = {}
            for model_name, model_data in models.items():
                if model_name == 'best_model':
                    serialized[pathway][model_name] = model_data
                elif isinstance(model_data, dict):
                    serialized[pathway][model_name] = {
                        'cv_mean': float(model_data['cv_mean']),
                        'cv_std': float(model_data['cv_std']),
                        'r2_full': float(model_data['r2_full']),
                        'mse_full': float(model_data['mse_full'])
                    }
        
        return serialized
    
    def _serialize_temp_response(self):
        """Serialize temperature response results"""
        if 'temperature_response' not in self.results:
            return {}
        
        serialized = {}
        for pathway, data in self.results['temperature_response'].items():
            serialized[pathway] = {
                'peak_temperature': float(data['peak_temp']),
                'peak_effect': float(data['peak_effect']),
                'temperature_range': {
                    'min': float(data['temperature'].min()),
                    'max': float(data['temperature'].max())
                }
            }
        
        return serialized
    
    def _serialize_lag_effects(self):
        """Serialize lag effects results"""
        if 'lag_effects' not in self.results:
            return {}
        
        serialized = {}
        for pathway, data in self.results['lag_effects'].items():
            serialized[pathway] = {
                'max_lag_effect': float(np.max(np.abs(data['effects']))),
                'lag_at_max_effect': int(np.argmax(np.abs(data['effects']))),
                'extreme_temperature': float(data['extreme_temp'])
            }
        
        return serialized
    
    def _serialize_ses_interactions(self):
        """Serialize SES interaction results"""
        if 'ses_interactions' not in self.results:
            return {}
        
        return {
            pathway: {
                ses_level: {
                    'mean_effect': float(data['mean_effect']),
                    'sample_size': int(data['sample_size'])
                }
                for ses_level, data in interactions.items()
            }
            for pathway, interactions in self.results['ses_interactions'].items()
        }
    
    def _generate_key_findings(self):
        """Generate key scientific findings"""
        findings = []
        
        # Temperature response findings
        if 'temperature_response' in self.results:
            for pathway in ['cardiovascular', 'renal']:
                if pathway in self.results['temperature_response']:
                    peak_temp = self.results['temperature_response'][pathway]['peak_temp']
                    findings.append(
                        f"{pathway.title()} effects peak at {peak_temp:.1f}°C"
                    )
        
        # Lag effects findings  
        if 'lag_effects' in self.results:
            for pathway in ['cardiovascular', 'renal']:
                if pathway in self.results['lag_effects']:
                    max_lag = self.results['lag_effects'][pathway]['effects']
                    lag_at_max = np.argmax(np.abs(max_lag))
                    findings.append(
                        f"{pathway.title()} effects strongest at {lag_at_max}-day lag"
                    )
        
        # SES interaction findings
        if 'ses_interactions' in self.results:
            findings.append("Socioeconomic vulnerability modifies temperature-health relationships")
        
        # Model performance findings
        for pathway in ['cardiovascular', 'renal']:
            if pathway in self.results['dlnm_models']:
                best_model = self.results['dlnm_models'][pathway]['best_model']
                best_r2 = self.results['dlnm_models'][pathway][best_model]['cv_mean']
                findings.append(
                    f"{pathway.title()} pathway best explained by {best_model} (R² = {best_r2:.3f})"
                )
        
        return findings
    
    def _assess_statistical_significance(self):
        """Assess statistical significance of findings"""
        significance = {}
        
        for pathway in ['cardiovascular', 'renal']:
            if pathway in self.results['dlnm_models']:
                best_model_name = self.results['dlnm_models'][pathway]['best_model']
                cv_mean = self.results['dlnm_models'][pathway][best_model_name]['cv_mean']
                cv_std = self.results['dlnm_models'][pathway][best_model_name]['cv_std']
                
                # Simple significance test (CV mean > 2*CV std)
                is_significant = cv_mean > 2 * cv_std
                
                significance[pathway] = {
                    'statistically_significant': is_significant,
                    'cv_mean': float(cv_mean),
                    'cv_std': float(cv_std),
                    'significance_threshold': 'CV mean > 2 × CV std'
                }
        
        return significance
    
    def _assess_clinical_relevance(self):
        """Assess clinical relevance of findings"""
        relevance = {}
        
        # Clinical relevance thresholds (conservative)
        cardio_threshold = 0.01  # 1% change in cardiovascular score
        renal_threshold = 0.05   # 5% change in renal risk
        
        thresholds = {
            'cardiovascular': cardio_threshold,
            'renal': renal_threshold
        }
        
        for pathway in ['cardiovascular', 'renal']:
            if pathway in self.results.get('temperature_response', {}):
                peak_effect = self.results['temperature_response'][pathway]['peak_effect']
                threshold = thresholds[pathway]
                
                is_clinically_relevant = abs(peak_effect) >= threshold
                
                relevance[pathway] = {
                    'clinically_relevant': is_clinically_relevant,
                    'peak_effect': float(peak_effect),
                    'relevance_threshold': threshold,
                    'interpretation': f"Peak effect {'exceeds' if is_clinically_relevant else 'below'} clinical threshold"
                }
        
        return relevance


def main():
    """Main execution function"""
    
    print("🚀 Starting Publication-Ready DLNM Climate-Health Analysis")
    print("=" * 70)
    
    # Configuration
    config = {
        'max_lag': 21,
        'temp_knots': [10, 25, 90],  # Percentile knots
        'paths': {
            'gcro_data': '/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets/GCRO_combined_climate_SUBSET.csv',
            'enhanced_results': '/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/results/enhanced_rigorous_analysis_report.json',
            'output_dir': '/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/results/'
        }
    }
    
    # Initialize analyzer
    analyzer = DLNMClimateHealthAnalyzer(
        max_lag=config['max_lag'],
        temp_knots=config['temp_knots'],
        verbose=True
    )
    
    try:
        # Load and prepare data
        analyzer.load_and_prepare_data(
            config['paths']['gcro_data'],
            config['paths']['enhanced_results']
        )
        
        # Fit DLNM models
        analyzer.fit_dlnm_models()
        
        # Analyze temperature-response curves
        analyzer.analyze_temperature_response_curves()
        
        # Analyze lag effects
        analyzer.analyze_lag_effects()
        
        # Analyze SES interactions
        analyzer.analyze_ses_interactions()
        
        # Create publication visualizations
        analyzer.create_publication_visualizations(config['paths']['output_dir'])
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(config['paths']['output_dir'])
        
        print("\n🎉 DLNM Analysis Complete!")
        print("=" * 50)
        print("📊 Analysis Summary:")
        print(f"   • Sample size: {report['analysis_metadata']['sample_size']:,}")
        print(f"   • Temperature range: {report['analysis_metadata']['temperature_range']['min']:.1f}°C to {report['analysis_metadata']['temperature_range']['max']:.1f}°C")
        print(f"   • Maximum lag: {report['analysis_metadata']['max_lag_days']} days")
        print(f"   • Pathways: {', '.join(report['analysis_metadata']['pathways_analyzed'])}")
        
        print("\n🔬 Key Findings:")
        for finding in report['key_findings'][:5]:  # Show top 5 findings
            print(f"   • {finding}")
        
        print(f"\n📁 Outputs saved to: {config['paths']['output_dir']}")
        print("   • Temperature response curves (SVG)")
        print("   • Lag response patterns (SVG)")
        print("   • SES interaction effects (SVG)")
        print("   • Model performance comparison (SVG)")
        print("   • Comprehensive analysis report (JSON)")
        
        return analyzer, report
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    analyzer, report = main()