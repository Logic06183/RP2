#!/usr/bin/env python3
"""
HEAT Analysis Pipeline - Module 3: Machine Learning Analysis
===========================================================

This module handles the machine learning pipeline including model training,
evaluation, and explainable AI (XAI) analysis using SHAP.

Author: Craig Parker
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from datetime import datetime

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Explainable AI
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeatMLAnalyzer:
    """
    Machine learning analyzer for climate-health relationships with explainable AI.
    """

    def __init__(self, results_dir: str = "../results"):
        """
        Initialize the ML analyzer.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.models = {}
        self.results = {}
        self.shap_values = {}
        self.feature_importance = {}

        # Configure plot style
        plt.style.use('default')
        sns.set_palette("husl")

    def prepare_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for temporal cross-validation to prevent data leakage.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame sorted by time with temporal index
        """
        logger.info("⏰ Preparing temporal data structure...")

        df_temporal = df.copy()

        # Try to create temporal sorting
        temporal_cols = ['primary_date', 'survey_year', 'year']
        temporal_col = None

        for col in temporal_cols:
            if col in df_temporal.columns:
                temporal_col = col
                break

        if temporal_col:
            if temporal_col == 'primary_date':
                df_temporal['temporal_sort'] = pd.to_datetime(df_temporal[temporal_col], errors='coerce')
            else:
                df_temporal['temporal_sort'] = pd.to_numeric(df_temporal[temporal_col], errors='coerce')

            # Sort by temporal column
            df_temporal = df_temporal.sort_values('temporal_sort')
            df_temporal = df_temporal.reset_index(drop=True)

            logger.info(f"   ✅ Data sorted by {temporal_col}")
        else:
            logger.warning("   ⚠️  No temporal column found, using original order")

        return df_temporal

    def train_biomarker_model(self,
                            df: pd.DataFrame,
                            target_col: str,
                            feature_cols: List[str],
                            cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train a Random Forest model for a specific biomarker.

        Args:
            df: Input DataFrame
            target_col: Target variable column name
            feature_cols: List of feature column names
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary containing model results
        """
        logger.info(f"🎯 Training model for: {target_col}")

        # Prepare data
        valid_data = df.dropna(subset=[target_col])
        if len(valid_data) == 0:
            logger.error(f"   ❌ No valid data for {target_col}")
            return {'error': 'No valid data'}

        # Select features and target
        available_features = [col for col in feature_cols if col in valid_data.columns]
        X = valid_data[available_features]
        y = valid_data[target_col]

        # Remove rows with any missing features
        complete_data = pd.concat([X, y], axis=1).dropna()
        if len(complete_data) == 0:
            logger.error(f"   ❌ No complete data for {target_col}")
            return {'error': 'No complete data after removing missing values'}

        X = complete_data[available_features]
        y = complete_data[target_col]

        logger.info(f"   📊 Training data: {len(X):,} samples, {len(available_features)} features")

        # Initialize model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation with temporal splits
        cv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        cv_predictions = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)

            cv_predictions.extend(list(zip(y_val.values, y_pred)))

            logger.info(f"   Fold {fold+1}/{cv_folds}: R² = {score:.4f}")

        # Train final model on all data
        model.fit(X, y)

        # Calculate final metrics
        y_pred_final = model.predict(X)
        final_r2 = r2_score(y, y_pred_final)
        final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))
        final_mae = mean_absolute_error(y, y_pred_final)

        # Store model and results
        self.models[target_col] = model

        results = {
            'target_variable': target_col,
            'n_samples': len(X),
            'n_features': len(available_features),
            'features_used': available_features,
            'cv_r2_scores': cv_scores,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae,
            'feature_importance': dict(zip(available_features, model.feature_importances_)),
            'model_params': model.get_params(),
            'cv_predictions': cv_predictions
        }

        logger.info(f"   ✅ Model trained: R² = {final_r2:.4f} (CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f})")

        return results

    def calculate_shap_values(self,
                            df: pd.DataFrame,
                            target_col: str,
                            feature_cols: List[str],
                            max_samples: int = 1000) -> Dict[str, Any]:
        """
        Calculate SHAP values for explainable AI analysis.

        Args:
            df: Input DataFrame
            target_col: Target variable column name
            feature_cols: List of feature column names
            max_samples: Maximum number of samples for SHAP calculation

        Returns:
            Dictionary containing SHAP analysis results
        """
        logger.info(f"🔍 Calculating SHAP values for: {target_col}")

        if target_col not in self.models:
            logger.error(f"   ❌ No trained model found for {target_col}")
            return {'error': 'No trained model'}

        # Prepare data
        valid_data = df.dropna(subset=[target_col])
        available_features = [col for col in feature_cols if col in valid_data.columns]
        X = valid_data[available_features]
        y = valid_data[target_col]

        # Remove rows with missing features
        complete_data = pd.concat([X, y], axis=1).dropna()
        X = complete_data[available_features]

        # Sample data for SHAP if too large
        if len(X) > max_samples:
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            logger.info(f"   📊 Using {max_samples:,} samples for SHAP calculation")
        else:
            X_sample = X
            logger.info(f"   📊 Using all {len(X):,} samples for SHAP calculation")

        # Get model
        model = self.models[target_col]

        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Calculate SHAP summary statistics
            shap_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_shap = dict(zip(available_features, shap_importance))

            # Sort by importance
            sorted_importance = sorted(feature_importance_shap.items(), key=lambda x: x[1], reverse=True)

            results = {
                'target_variable': target_col,
                'n_samples_shap': len(X_sample),
                'shap_values': shap_values,
                'feature_names': available_features,
                'shap_importance': feature_importance_shap,
                'top_features': sorted_importance[:10],  # Top 10 features
                'explainer': explainer
            }

            self.shap_values[target_col] = results

            logger.info(f"   ✅ SHAP values calculated")
            logger.info(f"   🔝 Top 3 features: {[feat[0] for feat in sorted_importance[:3]]}")

            return results

        except Exception as e:
            logger.error(f"   ❌ SHAP calculation failed: {e}")
            return {'error': str(e)}

    def analyze_all_biomarkers(self,
                             df: pd.DataFrame,
                             target_variables: List[str],
                             feature_cols: List[str]) -> Dict[str, Any]:
        """
        Analyze all biomarkers with ML and XAI.

        Args:
            df: Input DataFrame
            target_variables: List of target variable names
            feature_cols: List of feature column names

        Returns:
            Comprehensive analysis results
        """
        logger.info("🚀 Starting comprehensive biomarker analysis...")

        # Prepare temporal data
        df_temporal = self.prepare_temporal_data(df)

        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df_temporal),
                'total_features': len(feature_cols),
                'target_variables': target_variables
            },
            'model_results': {},
            'shap_results': {},
            'summary_statistics': {}
        }

        # Train models for each biomarker
        for target_col in target_variables:
            if target_col in df_temporal.columns:
                logger.info(f"\n{'='*60}")
                logger.info(f"🧬 ANALYZING: {target_col}")
                logger.info(f"{'='*60}")

                # Train model
                model_results = self.train_biomarker_model(
                    df_temporal, target_col, feature_cols, cv_folds=5
                )

                if 'error' not in model_results:
                    analysis_results['model_results'][target_col] = model_results

                    # Calculate SHAP values
                    shap_results = self.calculate_shap_values(
                        df_temporal, target_col, feature_cols, max_samples=1000
                    )

                    if 'error' not in shap_results:
                        analysis_results['shap_results'][target_col] = {
                            'target_variable': shap_results['target_variable'],
                            'n_samples_shap': shap_results['n_samples_shap'],
                            'shap_importance': shap_results['shap_importance'],
                            'top_features': shap_results['top_features']
                        }

                else:
                    logger.error(f"   ❌ Skipping {target_col} due to error: {model_results.get('error', 'Unknown')}")

        # Generate summary statistics
        self._generate_summary_statistics(analysis_results)

        # Save results
        self.save_results(analysis_results)

        logger.info("\n" + "="*60)
        logger.info("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        logger.info("="*60)

        return analysis_results

    def _generate_summary_statistics(self, analysis_results: Dict[str, Any]):
        """Generate summary statistics for the analysis."""
        model_results = analysis_results['model_results']

        if model_results:
            # Performance summary
            performance_summary = {}
            for target, results in model_results.items():
                performance_summary[target] = {
                    'r2_score': results['final_r2'],
                    'cv_r2_mean': results['cv_r2_mean'],
                    'cv_r2_std': results['cv_r2_std'],
                    'n_samples': results['n_samples'],
                    'performance_tier': self._classify_performance(results['final_r2'])
                }

            # Sort by R² score
            sorted_performance = sorted(
                performance_summary.items(),
                key=lambda x: x[1]['r2_score'],
                reverse=True
            )

            analysis_results['summary_statistics'] = {
                'n_successful_models': len(model_results),
                'best_performing_biomarker': sorted_performance[0][0] if sorted_performance else None,
                'performance_summary': performance_summary,
                'average_r2': np.mean([r['final_r2'] for r in model_results.values()]),
                'models_above_threshold': len([r for r in model_results.values() if r['final_r2'] > 0.5])
            }

    def _classify_performance(self, r2_score: float) -> str:
        """Classify model performance based on R² score."""
        if r2_score >= 0.7:
            return "Excellent"
        elif r2_score >= 0.6:
            return "Good"
        elif r2_score >= 0.5:
            return "Moderate"
        elif r2_score >= 0.3:
            return "Weak"
        else:
            return "Poor"

    def create_summary_visualizations(self, analysis_results: Dict[str, Any]):
        """Create summary visualizations of the analysis."""
        logger.info("📊 Creating summary visualizations...")

        model_results = analysis_results['model_results']
        if not model_results:
            logger.warning("   ⚠️  No model results to visualize")
            return

        # Performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # R² scores comparison
        biomarkers = list(model_results.keys())
        r2_scores = [model_results[bio]['final_r2'] for bio in biomarkers]
        cv_means = [model_results[bio]['cv_r2_mean'] for bio in biomarkers]
        cv_stds = [model_results[bio]['cv_r2_std'] for bio in biomarkers]

        x_pos = np.arange(len(biomarkers))
        ax1.bar(x_pos, r2_scores, alpha=0.7, label='Final R²')
        ax1.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', color='red', label='CV R² (mean ± std)')
        ax1.set_xlabel('Biomarkers')
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([bio.split('(')[0].strip() for bio in biomarkers], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Sample sizes
        sample_sizes = [model_results[bio]['n_samples'] for bio in biomarkers]
        ax2.bar(x_pos, sample_sizes, alpha=0.7, color='green')
        ax2.set_xlabel('Biomarkers')
        ax2.set_ylabel('Sample Size')
        ax2.set_title('Sample Sizes by Biomarker')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([bio.split('(')[0].strip() for bio in biomarkers], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"   ✅ Summary visualizations saved to {self.results_dir}")

    def save_results(self, analysis_results: Dict[str, Any]):
        """Save analysis results to files."""
        logger.info("💾 Saving analysis results...")

        # Save main results (excluding large objects)
        results_to_save = analysis_results.copy()

        # Remove large objects that can't be JSON serialized
        if 'shap_results' in results_to_save:
            for target in results_to_save['shap_results']:
                if 'explainer' in results_to_save['shap_results'][target]:
                    del results_to_save['shap_results'][target]['explainer']

        # Save JSON results
        with open(self.results_dir / 'analysis_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        # Save summary table
        if analysis_results['model_results']:
            summary_data = []
            for target, results in analysis_results['model_results'].items():
                summary_data.append({
                    'Biomarker': target,
                    'R² Score': f"{results['final_r2']:.4f}",
                    'CV R² (mean ± std)': f"{results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}",
                    'Sample Size': results['n_samples'],
                    'RMSE': f"{results['final_rmse']:.4f}",
                    'Performance': self._classify_performance(results['final_r2'])
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('R² Score', ascending=False)
            summary_df.to_csv(self.results_dir / 'model_performance_summary.csv', index=False)

        logger.info(f"   ✅ Results saved to {self.results_dir}")


def main():
    """
    Example usage of the ML analyzer.
    """
    # This would typically be called after preprocessing
    from data_loader import HeatDataLoader
    from data_preprocessor import HeatDataPreprocessor

    # Load and preprocess data
    loader = HeatDataLoader(data_dir="../data")
    df = loader.load_master_dataset(sample_fraction=1.0)  # NEVER sample!
    clean_df = loader.get_clean_dataset()

    preprocessor = HeatDataPreprocessor()
    processed_df, preprocessing_report = preprocessor.preprocess_complete_pipeline(clean_df)

    # Extract features and targets
    target_variables = preprocessing_report['target_variables']
    feature_cols = [col for col in processed_df.columns if col not in target_variables]

    # Initialize analyzer
    analyzer = HeatMLAnalyzer(results_dir="../results")

    # Run comprehensive analysis
    analysis_results = analyzer.analyze_all_biomarkers(
        processed_df, target_variables, feature_cols
    )

    # Create visualizations
    analyzer.create_summary_visualizations(analysis_results)

    return analysis_results


if __name__ == "__main__":
    analysis_results = main()