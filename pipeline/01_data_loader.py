#!/usr/bin/env python3
"""
HEAT Analysis Pipeline - Module 1: Data Loading
===============================================

This module handles loading and initial validation of the integrated dataset.
It ensures consistent data types and provides summary statistics.

Author: Craig Parker
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeatDataLoader:
    """
    Loads and validates the HEAT integrated dataset with proper data type handling.
    """

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.master_dataset_path = self.data_dir / "MASTER_INTEGRATED_DATASET.csv"
        self.df = None
        self.validation_report = {}

    def load_master_dataset(self,
                          sample_fraction: float = 1.0,
                          low_memory: bool = False) -> pd.DataFrame:
        """
        Load the master integrated dataset with proper data type handling.

        Args:
            sample_fraction: Fraction of data to load (1.0 = all data, NEVER sample for analysis!)
            low_memory: Whether to use low memory mode (False = faster, True = less memory)

        Returns:
            Loaded and validated DataFrame
        """
        if sample_fraction < 1.0:
            logger.warning(f"⚠️  SAMPLING DETECTED: Loading only {sample_fraction*100:.1f}% of data!")
            logger.warning("🚨 CRITICAL: For publication analysis, ALWAYS use sample_fraction=1.0")

        logger.info(f"Loading master dataset from: {self.master_dataset_path}")

        # Define data types for key columns to avoid mixed type warnings
        dtype_dict = {
            'gcro_id': 'str',
            'survey_wave': 'str',
            'data_source': 'str',
            'anonymous_patient_id': 'str',
            'study_source': 'str',
            'city': 'str',
            'province': 'str',
            'country': 'str'
        }

        # Load with proper handling
        self.df = pd.read_csv(
            self.master_dataset_path,
            dtype=dtype_dict,
            low_memory=low_memory
        )

        # Sample if requested (but warn against it)
        if sample_fraction < 1.0:
            n_sample = int(len(self.df) * sample_fraction)
            self.df = self.df.sample(n=n_sample, random_state=42).reset_index(drop=True)
            logger.warning(f"⚠️  Dataset sampled to {len(self.df):,} records")

        logger.info(f"✅ Loaded dataset: {len(self.df):,} records × {len(self.df.columns)} columns")

        return self.df

    def validate_data_structure(self) -> Dict:
        """
        Validate the data structure and create a comprehensive report.

        Returns:
            Dictionary containing validation results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_master_dataset() first.")

        logger.info("🔍 Validating data structure...")

        # Basic structure validation
        self.validation_report = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'data_sources': self.df['data_source'].value_counts().to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }

        # Temporal coverage validation
        self._validate_temporal_coverage()

        # Geographic coverage validation
        self._validate_geographic_coverage()

        # Clinical data validation
        self._validate_clinical_data()

        # Climate data validation
        self._validate_climate_data()

        # GCRO data validation
        self._validate_gcro_data()

        logger.info("✅ Data structure validation complete")
        return self.validation_report

    def _validate_temporal_coverage(self):
        """Validate temporal data coverage."""
        # GCRO temporal coverage
        gcro_data = self.df[self.df['data_source'] == 'GCRO']
        if len(gcro_data) > 0:
            gcro_years = sorted(gcro_data['survey_year'].dropna().unique())
            self.validation_report['gcro_survey_years'] = [int(y) for y in gcro_years if pd.notna(y)]

        # Clinical temporal coverage
        clinical_data = self.df[self.df['data_source'] == 'RP2_Clinical']
        if len(clinical_data) > 0:
            # Convert dates safely
            clinical_dates = pd.to_datetime(clinical_data['primary_date'], errors='coerce')
            valid_dates = clinical_dates.dropna()
            if len(valid_dates) > 0:
                self.validation_report['clinical_date_range'] = {
                    'start': str(valid_dates.min().date()),
                    'end': str(valid_dates.max().date()),
                    'valid_dates': len(valid_dates),
                    'invalid_dates': len(clinical_dates) - len(valid_dates)
                }

    def _validate_geographic_coverage(self):
        """Validate geographic data coverage."""
        # Convert coordinates to numeric, handling mixed types
        lat_col = 'latitude'
        lon_col = 'longitude'

        if lat_col in self.df.columns and lon_col in self.df.columns:
            # Convert to numeric, coercing errors to NaN
            lats = pd.to_numeric(self.df[lat_col], errors='coerce')
            lons = pd.to_numeric(self.df[lon_col], errors='coerce')

            # Filter to valid coordinates
            valid_coords = (lats.notna()) & (lons.notna())

            if valid_coords.sum() > 0:
                self.validation_report['geographic_coverage'] = {
                    'latitude_range': [float(lats[valid_coords].min()), float(lats[valid_coords].max())],
                    'longitude_range': [float(lons[valid_coords].min()), float(lons[valid_coords].max())],
                    'valid_coordinates': int(valid_coords.sum()),
                    'missing_coordinates': int((~valid_coords).sum())
                }
            else:
                self.validation_report['geographic_coverage'] = {
                    'error': 'No valid coordinates found'
                }

    def _validate_clinical_data(self):
        """Validate clinical biomarker data."""
        clinical_data = self.df[self.df['data_source'] == 'RP2_Clinical']

        if len(clinical_data) > 0:
            # Key biomarkers to check
            biomarkers = [
                'CD4 cell count (cells/µL)',
                'Hemoglobin (g/dL)',
                'Creatinine (mg/dL)',
                'FASTING GLUCOSE',
                'FASTING TOTAL CHOLESTEROL'
            ]

            biomarker_stats = {}
            for biomarker in biomarkers:
                if biomarker in clinical_data.columns:
                    values = pd.to_numeric(clinical_data[biomarker], errors='coerce')
                    biomarker_stats[biomarker] = {
                        'total_records': len(clinical_data),
                        'non_null_count': int(values.notna().sum()),
                        'completion_rate': float(values.notna().sum() / len(clinical_data)),
                        'mean': float(values.mean()) if values.notna().sum() > 0 else None,
                        'std': float(values.std()) if values.notna().sum() > 0 else None
                    }

            self.validation_report['clinical_biomarkers'] = biomarker_stats
        else:
            self.validation_report['clinical_biomarkers'] = {'error': 'No clinical data found'}

    def _validate_climate_data(self):
        """Validate climate data integration."""
        # Identify ERA5 climate columns
        era5_cols = [col for col in self.df.columns if 'era5' in col.lower()]

        climate_stats = {}
        for col in era5_cols:
            values = pd.to_numeric(self.df[col], errors='coerce')
            climate_stats[col] = {
                'non_null_count': int(values.notna().sum()),
                'completion_rate': float(values.notna().sum() / len(self.df)),
                'mean': float(values.mean()) if values.notna().sum() > 0 else None
            }

        self.validation_report['climate_data'] = {
            'era5_columns_count': len(era5_cols),
            'column_stats': climate_stats
        }

    def _validate_gcro_data(self):
        """Validate GCRO socioeconomic data."""
        gcro_data = self.df[self.df['data_source'] == 'GCRO']

        if len(gcro_data) > 0:
            # Check key GCRO variables
            gcro_vars = ['age', 'sex', 'race', 'education', 'employment', 'income']
            gcro_stats = {}

            for var in gcro_vars:
                if var in gcro_data.columns:
                    non_null = gcro_data[var].notna().sum()
                    gcro_stats[var] = {
                        'non_null_count': int(non_null),
                        'completion_rate': float(non_null / len(gcro_data))
                    }

            self.validation_report['gcro_data'] = {
                'total_gcro_records': len(gcro_data),
                'variable_stats': gcro_stats
            }
        else:
            self.validation_report['gcro_data'] = {'error': 'No GCRO data found'}

    def print_validation_summary(self):
        """Print a comprehensive validation summary."""
        if not self.validation_report:
            logger.error("No validation report available. Run validate_data_structure() first.")
            return

        print("\n" + "="*60)
        print("🏥 HEAT DATASET VALIDATION SUMMARY")
        print("="*60)

        print(f"\n📊 DATASET SCALE:")
        print(f"   Total Records: {self.validation_report['total_records']:,}")
        print(f"   Total Columns: {self.validation_report['total_columns']}")
        print(f"   Memory Usage: {self.validation_report['memory_usage_mb']:.1f} MB")

        print(f"\n🔍 DATA SOURCES:")
        for source, count in self.validation_report['data_sources'].items():
            percentage = count / self.validation_report['total_records'] * 100
            print(f"   {source}: {count:,} records ({percentage:.1f}%)")

        if 'gcro_survey_years' in self.validation_report:
            print(f"\n📅 TEMPORAL COVERAGE:")
            print(f"   GCRO Survey Years: {self.validation_report['gcro_survey_years']}")

        if 'clinical_date_range' in self.validation_report:
            cdr = self.validation_report['clinical_date_range']
            print(f"   Clinical Date Range: {cdr['start']} to {cdr['end']}")
            print(f"   Valid Clinical Dates: {cdr['valid_dates']:,}")

        if 'geographic_coverage' in self.validation_report and 'error' not in self.validation_report['geographic_coverage']:
            geo = self.validation_report['geographic_coverage']
            print(f"\n🌍 GEOGRAPHIC COVERAGE:")
            print(f"   Latitude Range: {geo['latitude_range'][0]:.3f} to {geo['latitude_range'][1]:.3f}")
            print(f"   Longitude Range: {geo['longitude_range'][0]:.3f} to {geo['longitude_range'][1]:.3f}")
            print(f"   Valid Coordinates: {geo['valid_coordinates']:,}")

        if 'clinical_biomarkers' in self.validation_report and 'error' not in self.validation_report['clinical_biomarkers']:
            print(f"\n🧬 CLINICAL BIOMARKERS:")
            for biomarker, stats in self.validation_report['clinical_biomarkers'].items():
                print(f"   {biomarker}: {stats['non_null_count']:,} values ({stats['completion_rate']*100:.1f}%)")

        if 'climate_data' in self.validation_report:
            cd = self.validation_report['climate_data']
            print(f"\n🌡️  CLIMATE DATA:")
            print(f"   ERA5 Columns: {cd['era5_columns_count']}")

            # Show top climate variables by completion
            if cd['column_stats']:
                sorted_climate = sorted(cd['column_stats'].items(),
                                      key=lambda x: x[1]['completion_rate'], reverse=True)
                print(f"   Top Climate Variables:")
                for col, stats in sorted_climate[:3]:
                    print(f"     {col}: {stats['completion_rate']*100:.1f}% complete")

        print("\n" + "="*60)

    def get_clean_dataset(self) -> pd.DataFrame:
        """
        Return the loaded dataset with basic cleaning applied.

        Returns:
            Cleaned DataFrame ready for preprocessing
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_master_dataset() first.")

        # Create a copy for cleaning
        clean_df = self.df.copy()

        # Convert coordinate columns to numeric
        for coord_col in ['latitude', 'longitude']:
            if coord_col in clean_df.columns:
                clean_df[coord_col] = pd.to_numeric(clean_df[coord_col], errors='coerce')

        # Convert date columns
        if 'primary_date' in clean_df.columns:
            clean_df['primary_date'] = pd.to_datetime(clean_df['primary_date'], errors='coerce')

        # Convert numeric columns that might be strings
        numeric_cols = [col for col in clean_df.columns if 'era5' in col.lower()]
        for col in numeric_cols:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

        logger.info(f"✅ Cleaned dataset ready: {len(clean_df):,} records")
        return clean_df


def main():
    """
    Example usage of the data loader.
    """
    # Initialize loader
    loader = HeatDataLoader(data_dir="../data")

    # Load full dataset (CRITICAL: Never sample for analysis!)
    df = loader.load_master_dataset(sample_fraction=1.0)

    # Validate data structure
    validation_report = loader.validate_data_structure()

    # Print summary
    loader.print_validation_summary()

    # Get cleaned dataset
    clean_df = loader.get_clean_dataset()

    return clean_df, validation_report


if __name__ == "__main__":
    clean_df, validation_report = main()