#!/usr/bin/env python3
"""
HEAT Analysis Pipeline - Module 2: Data Preprocessing
=====================================================

This module handles data preprocessing including feature engineering,
missing value handling, and preparation for machine learning.

Author: Craig Parker
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeatDataPreprocessor:
    """
    Preprocesses HEAT data for machine learning analysis with focus on climate-health relationships.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = {}
        self.preprocessing_report = {}

    def identify_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize column types for appropriate preprocessing.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with categorized column names
        """
        logger.info("🔍 Identifying data types and feature categories...")

        categories = {
            'identifiers': [],
            'temporal': [],
            'geographic': [],
            'demographic': [],
            'socioeconomic': [],
            'clinical_biomarkers': [],
            'climate_features': [],
            'categorical': [],
            'numerical': [],
            'exclude': []
        }

        for col in df.columns:
            col_lower = col.lower()

            # Identifiers (exclude from analysis)
            if any(id_term in col_lower for id_term in ['id', 'index', 'source', 'patient']):
                categories['identifiers'].append(col)

            # Temporal features
            elif any(temp_term in col_lower for temp_term in ['date', 'year', 'month', 'week', 'time']):
                categories['temporal'].append(col)

            # Geographic features
            elif any(geo_term in col_lower for geo_term in ['latitude', 'longitude', 'ward', 'municipality', 'city', 'province', 'country']):
                categories['geographic'].append(col)

            # Climate features (ERA5, temperature, etc.)
            elif any(climate_term in col_lower for climate_term in ['era5', 'temp', 'climate', 'heat', 'weather']):
                categories['climate_features'].append(col)

            # Clinical biomarkers
            elif any(bio_term in col_lower for bio_term in [
                'cd4', 'viral load', 'hemoglobin', 'creatinine', 'glucose', 'cholesterol',
                'albumin', 'bilirubin', 'pressure', 'heart rate', 'weight', 'height'
            ]):
                categories['clinical_biomarkers'].append(col)

            # Demographic
            elif any(demo_term in col_lower for demo_term in ['age', 'sex', 'race', 'gender']):
                categories['demographic'].append(col)

            # Socioeconomic
            elif any(se_term in col_lower for se_term in ['income', 'education', 'employment', 'housing']):
                categories['socioeconomic'].append(col)

            # Categorical vs numerical
            elif df[col].dtype == 'object' or df[col].nunique() < 10:
                categories['categorical'].append(col)
            else:
                categories['numerical'].append(col)

        # Remove overlaps (features can be in multiple categories)
        self.feature_categories = categories

        # Log summary
        for category, cols in categories.items():
            if cols:
                logger.info(f"   {category}: {len(cols)} columns")

        return categories

    def engineer_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional climate features from ERA5 data.

        Args:
            df: Input DataFrame with ERA5 features

        Returns:
            DataFrame with additional climate features
        """
        logger.info("🌡️  Engineering climate features...")

        df_enhanced = df.copy()

        # Identify temperature columns
        temp_cols = [col for col in df.columns if 'era5_temp' in col.lower()]

        if temp_cols:
            # Temperature variability features
            mean_cols = [col for col in temp_cols if 'mean' in col]
            max_cols = [col for col in temp_cols if 'max' in col]

            if len(mean_cols) >= 2:
                # Temperature variability across time windows
                df_enhanced['temp_variability_7d_vs_1d'] = (
                    df_enhanced.get('era5_temp_7d_mean', 0) - df_enhanced.get('era5_temp_1d_mean', 0)
                )
                df_enhanced['temp_variability_30d_vs_7d'] = (
                    df_enhanced.get('era5_temp_30d_mean', 0) - df_enhanced.get('era5_temp_7d_mean', 0)
                )

            if len(max_cols) >= 2:
                # Maximum temperature differences
                df_enhanced['temp_max_variability_7d_vs_1d'] = (
                    df_enhanced.get('era5_temp_7d_max', 0) - df_enhanced.get('era5_temp_1d_max', 0)
                )

            # Heat stress indicators
            if 'era5_temp_1d_mean' in df_enhanced.columns:
                # Simple heat stress threshold (>25°C)
                df_enhanced['heat_stress_indicator'] = (df_enhanced['era5_temp_1d_mean'] > 25).astype(int)

                # Temperature quintiles
                temp_values = df_enhanced['era5_temp_1d_mean'].dropna()
                if len(temp_values) > 0:
                    quintiles = temp_values.quantile([0.2, 0.4, 0.6, 0.8])
                    df_enhanced['temp_quintile'] = pd.cut(
                        df_enhanced['era5_temp_1d_mean'],
                        bins=[-np.inf] + quintiles.tolist() + [np.inf],
                        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
                    )

            logger.info(f"   ✅ Added {len([col for col in df_enhanced.columns if col not in df.columns])} climate features")

        return df_enhanced

    def preprocess_clinical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess clinical biomarker data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with preprocessed clinical data
        """
        logger.info("🧬 Preprocessing clinical biomarker data...")

        df_processed = df.copy()
        clinical_cols = self.feature_categories.get('clinical_biomarkers', [])

        for col in clinical_cols:
            if col in df_processed.columns:
                # Convert to numeric
                values = pd.to_numeric(df_processed[col], errors='coerce')

                # Remove extreme outliers (beyond 5 standard deviations)
                mean_val = values.mean()
                std_val = values.std()
                if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                    outlier_threshold = 5 * std_val
                    outliers = np.abs(values - mean_val) > outlier_threshold
                    values[outliers] = np.nan

                df_processed[col] = values

        logger.info(f"   ✅ Processed {len(clinical_cols)} clinical variables")
        return df_processed

    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for machine learning.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("🏷️  Encoding categorical variables...")

        df_encoded = df.copy()
        categorical_cols = self.feature_categories.get('categorical', [])

        for col in categorical_cols:
            if col in df_encoded.columns and col not in self.feature_categories.get('identifiers', []):
                # Handle missing values first
                df_encoded[col] = df_encoded[col].fillna('Unknown')

                # Use label encoding for now (could be enhanced with one-hot encoding)
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()

                try:
                    df_encoded[col + '_encoded'] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                    logger.info(f"   ✅ Encoded {col}: {len(self.encoders[col].classes_)} unique values")
                except Exception as e:
                    logger.warning(f"   ⚠️  Failed to encode {col}: {e}")

        return df_encoded

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'adaptive') -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.

        Args:
            df: Input DataFrame
            strategy: Strategy for imputation ('adaptive', 'drop', 'mean', 'median')

        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"🔧 Handling missing values with '{strategy}' strategy...")

        df_imputed = df.copy()

        if strategy == 'adaptive':
            # Different strategies for different types of features
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
            categorical_cols = df_imputed.select_dtypes(include=['object']).columns

            # Impute numeric columns with median
            for col in numeric_cols:
                if col not in self.feature_categories.get('identifiers', []):
                    missing_pct = df_imputed[col].isna().sum() / len(df_imputed)
                    if missing_pct > 0:
                        if missing_pct > 0.5:
                            logger.warning(f"   ⚠️  {col}: {missing_pct*100:.1f}% missing (high missingness)")

                        imputer_key = f"{col}_numeric"
                        if imputer_key not in self.imputers:
                            self.imputers[imputer_key] = SimpleImputer(strategy='median')

                        values_reshaped = df_imputed[[col]].values
                        df_imputed[col] = self.imputers[imputer_key].fit_transform(values_reshaped).flatten()

            # Impute categorical columns with mode
            for col in categorical_cols:
                if col not in self.feature_categories.get('identifiers', []):
                    missing_pct = df_imputed[col].isna().sum() / len(df_imputed)
                    if missing_pct > 0:
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode().iloc[0] if len(df_imputed[col].mode()) > 0 else 'Unknown')

        elif strategy == 'drop':
            # Drop rows with any missing values
            initial_rows = len(df_imputed)
            df_imputed = df_imputed.dropna()
            logger.info(f"   ✅ Dropped {initial_rows - len(df_imputed):,} rows with missing values")

        logger.info(f"   ✅ Missing value handling complete")
        return df_imputed

    def create_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create final feature set for machine learning.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (feature DataFrame, target variable names)
        """
        logger.info("🎯 Creating ML feature set...")

        # Define target variables (key biomarkers for analysis)
        target_variables = [
            'CD4 cell count (cells/µL)',
            'Hemoglobin (g/dL)',
            'Creatinine (mg/dL)',
            'FASTING GLUCOSE',
            'FASTING TOTAL CHOLESTEROL'
        ]

        # Select feature columns (exclude identifiers and targets)
        exclude_patterns = ['id', 'index', 'source', 'patient', 'date', 'time']
        feature_cols = []

        for col in df.columns:
            col_lower = col.lower()
            # Include if it's not an identifier/temporal and not a target
            if (not any(pattern in col_lower for pattern in exclude_patterns) and
                col not in target_variables):
                feature_cols.append(col)

        # Focus on key feature types for climate-health analysis
        priority_features = []

        # Climate features (highest priority)
        climate_features = [col for col in feature_cols if any(term in col.lower() for term in ['era5', 'temp', 'climate', 'heat'])]
        priority_features.extend(climate_features)

        # Demographic features
        demo_features = [col for col in feature_cols if any(term in col.lower() for term in ['age', 'sex', 'race'])]
        priority_features.extend(demo_features)

        # Geographic features
        geo_features = [col for col in feature_cols if any(term in col.lower() for term in ['latitude', 'longitude'])]
        priority_features.extend(geo_features)

        # Socioeconomic features
        se_features = [col for col in feature_cols if any(term in col.lower() for term in ['income', 'education', 'employment'])]
        priority_features.extend(se_features)

        # Remove duplicates while preserving order
        final_features = list(dict.fromkeys(priority_features))

        # Create feature DataFrame
        feature_df = df[final_features + target_variables].copy()

        logger.info(f"   ✅ Created ML dataset: {len(final_features)} features, {len(target_variables)} targets")
        logger.info(f"   📊 Feature breakdown:")
        logger.info(f"      Climate features: {len(climate_features)}")
        logger.info(f"      Demographic features: {len(demo_features)}")
        logger.info(f"      Geographic features: {len(geo_features)}")
        logger.info(f"      Socioeconomic features: {len(se_features)}")

        return feature_df, target_variables

    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Scale numerical features for machine learning.

        Args:
            df: Input DataFrame
            feature_cols: List of feature column names to scale

        Returns:
            DataFrame with scaled features
        """
        logger.info("📏 Scaling numerical features...")

        df_scaled = df.copy()

        # Identify numerical columns to scale
        numeric_features = []
        for col in feature_cols:
            if col in df_scaled.columns and pd.api.types.is_numeric_dtype(df_scaled[col]):
                numeric_features.append(col)

        if numeric_features:
            # Initialize scaler
            scaler_key = 'main_features'
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()

            # Scale features
            scaled_values = self.scalers[scaler_key].fit_transform(df_scaled[numeric_features])
            df_scaled[numeric_features] = scaled_values

            logger.info(f"   ✅ Scaled {len(numeric_features)} numerical features")

        return df_scaled

    def preprocess_complete_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: Raw input DataFrame

        Returns:
            Tuple of (processed DataFrame, preprocessing report)
        """
        logger.info("🚀 Starting complete preprocessing pipeline...")

        # Step 1: Identify data types
        self.identify_data_types(df)

        # Step 2: Engineer climate features
        df_enhanced = self.engineer_climate_features(df)

        # Step 3: Preprocess clinical data
        df_clinical = self.preprocess_clinical_data(df_enhanced)

        # Step 4: Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_clinical)

        # Step 5: Handle missing values
        df_imputed = self.handle_missing_values(df_encoded, strategy='adaptive')

        # Step 6: Create ML features
        feature_df, target_variables = self.create_ml_features(df_imputed)

        # Step 7: Scale features (separate from targets)
        feature_cols = [col for col in feature_df.columns if col not in target_variables]
        df_final = self.scale_features(feature_df, feature_cols)

        # Create preprocessing report
        self.preprocessing_report = {
            'original_shape': df.shape,
            'final_shape': df_final.shape,
            'feature_categories': self.feature_categories,
            'target_variables': target_variables,
            'features_engineered': len([col for col in df_enhanced.columns if col not in df.columns]),
            'missing_value_strategy': 'adaptive',
            'scaling_applied': True
        }

        logger.info("✅ Preprocessing pipeline complete!")
        logger.info(f"   📊 Final dataset: {df_final.shape[0]:,} rows × {df_final.shape[1]} columns")

        return df_final, self.preprocessing_report


def main():
    """
    Example usage of the preprocessor.
    """
    # This would typically be called after loading data
    from data_loader import HeatDataLoader

    # Load data
    loader = HeatDataLoader(data_dir="../data")
    df = loader.load_master_dataset(sample_fraction=1.0)  # NEVER sample!
    clean_df = loader.get_clean_dataset()

    # Preprocess
    preprocessor = HeatDataPreprocessor()
    processed_df, report = preprocessor.preprocess_complete_pipeline(clean_df)

    print("\n" + "="*60)
    print("🔧 PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original shape: {report['original_shape']}")
    print(f"Final shape: {report['final_shape']}")
    print(f"Features engineered: {report['features_engineered']}")
    print(f"Target variables: {len(report['target_variables'])}")
    print("="*60)

    return processed_df, report


if __name__ == "__main__":
    processed_df, report = main()