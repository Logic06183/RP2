#!/usr/bin/env python3
"""
Enhanced Climate Data Integrator with Fixed Timestamp Handling
==============================================================

This script properly integrates climate data from zarr files with health data,
fixing the timestamp conversion issues and adding advanced temporal features.

Author: HEAT Research Team
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import xarray as xr
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EnhancedClimateIntegrator:
    """Enhanced climate data integrator with proper timestamp handling."""

    def __init__(self, climate_data_dir: str = "/home/cparker/selected_data_all/data/RP2_subsets/JHB"):
        """Initialize the climate data integrator."""
        self.climate_data_dir = Path(climate_data_dir)
        self.climate_datasets = {}
        self.integration_report = {}
        self.available_variables = {}

    def load_climate_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, xr.Dataset]:
        """Load specified climate datasets."""
        logger.info("🌡️  Loading climate datasets...")

        if dataset_names is None:
            # Default to most reliable datasets based on exploration
            dataset_names = [
                'ERA5_tas_native',      # Air temperature
                'ERA5_lst_native',      # Land surface temperature
                'ERA5_ws_native',       # Wind speed
                'ERA5-Land_tas_native', # High-res air temperature
                'SAAQIS_with_climate_variables'  # Integrated climate variables
            ]

        for dataset_name in dataset_names:
            file_path = self.climate_data_dir / f"{dataset_name}.zarr"

            if file_path.exists():
                try:
                    logger.info(f"  Loading {dataset_name}...")
                    ds = xr.open_zarr(file_path, chunks='auto')
                    self.climate_datasets[dataset_name] = ds

                    # Store available variables
                    self.available_variables[dataset_name] = list(ds.data_vars.keys())
                    logger.info(f"    Variables: {self.available_variables[dataset_name]}")

                except Exception as e:
                    logger.error(f"    ❌ Failed to load {dataset_name}: {e}")
            else:
                logger.warning(f"  ⚠️  Dataset not found: {dataset_name}")

        logger.info(f"✅ Loaded {len(self.climate_datasets)} climate datasets")
        return self.climate_datasets

    def extract_point_data_fixed(self,
                                 dataset: xr.Dataset,
                                 lat: float,
                                 lon: float,
                                 date: pd.Timestamp,
                                 tolerance_days: int = 1,
                                 variables: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Extract climate data for a specific point and time with FIXED timestamp handling.
        """
        try:
            # Find nearest spatial point
            point_data = None

            # Try standard coordinate names
            if 'latitude' in dataset.dims and 'longitude' in dataset.dims:
                point_data = dataset.sel(latitude=lat, longitude=lon, method='nearest')
            elif 'lat' in dataset.dims and 'lon' in dataset.dims:
                point_data = dataset.sel(lat=lat, lon=lon, method='nearest')
            else:
                # Try to find any lat/lon coordinate names
                coord_names = list(dataset.dims.keys())
                lat_coord = next((name for name in coord_names if 'lat' in name.lower()), None)
                lon_coord = next((name for name in coord_names if 'lon' in name.lower()), None)

                if lat_coord and lon_coord:
                    point_data = dataset.sel({lat_coord: lat, lon_coord: lon}, method='nearest')
                else:
                    return {}

            # Handle temporal selection with proper type conversion
            if 'time' in point_data.dims and len(point_data.time) > 0:
                # Convert target date to numpy datetime64 for comparison
                target_time_np = np.datetime64(date, 'ns')

                # Get time coordinate as numpy array
                time_array = point_data.time.values

                # Calculate time differences properly
                if len(time_array) > 0:
                    # Convert both to same type for comparison
                    time_diffs = np.abs(time_array.astype('datetime64[ns]') - target_time_np)

                    # Convert to days
                    time_diffs_days = time_diffs / np.timedelta64(1, 'D')

                    # Find minimum difference
                    min_diff_idx = np.argmin(time_diffs_days)
                    min_diff_days = time_diffs_days[min_diff_idx]

                    if min_diff_days <= tolerance_days:
                        # Select the nearest time point
                        nearest_time = time_array[min_diff_idx]
                        nearest_time_data = point_data.sel(time=nearest_time)
                    else:
                        return {}
                else:
                    return {}
            else:
                # No time dimension, use the data as is
                nearest_time_data = point_data

            # Extract values for specified variables or all variables
            extracted_data = {}
            vars_to_extract = variables if variables else list(nearest_time_data.data_vars.keys())

            for var_name in vars_to_extract:
                if var_name in nearest_time_data.data_vars:
                    try:
                        # Get the value and handle different array types
                        value = nearest_time_data[var_name].values

                        # Handle scalar or array values
                        if np.isscalar(value):
                            final_value = float(value)
                        else:
                            # If it's an array, take the first element
                            final_value = float(value.flat[0]) if value.size > 0 else np.nan

                        if not np.isnan(final_value):
                            extracted_data[var_name] = final_value

                    except Exception as e:
                        logger.debug(f"      Could not extract {var_name}: {e}")

            return extracted_data

        except Exception as e:
            logger.debug(f"    Point extraction error: {e}")
            return {}

    def create_temporal_features(self,
                                df: pd.DataFrame,
                                climate_cols: List[str],
                                lag_days: List[int] = [1, 3, 7],
                                rolling_windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """Create temporal features from climate variables."""
        logger.info("📈 Creating temporal features...")

        df_temporal = df.copy()

        # Find the ID column
        id_col = 'patient_id' if 'patient_id' in df.columns else 'gcro_id' if 'gcro_id' in df.columns else df.columns[0]
        date_col = 'primary_date' if 'primary_date' in df.columns else 'date'

        # Sort by date for proper temporal operations
        if date_col in df.columns:
            df_temporal = df_temporal.sort_values(date_col)

        # Skip temporal features for cross-sectional GCRO data (no longitudinal visits)
        if 'gcro_id' in df.columns and 'patient_id' not in df.columns:
            logger.info("  Skipping temporal features for cross-sectional GCRO data")
            return df_temporal

        for col in climate_cols:
            if col in df_temporal.columns:
                # Create lag features
                for lag in lag_days:
                    lag_col = f"{col}_lag_{lag}d"
                    if id_col in df_temporal.columns:
                        df_temporal[lag_col] = df_temporal.groupby(id_col)[col].shift(lag)
                    else:
                        df_temporal[lag_col] = df_temporal[col].shift(lag)
                    logger.info(f"  Created lag feature: {lag_col}")

                # Create rolling statistics
                for window in rolling_windows:
                    # Rolling mean
                    mean_col = f"{col}_mean_{window}d"
                    if id_col in df_temporal.columns:
                        df_temporal[mean_col] = df_temporal.groupby(id_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).mean()
                        )
                    else:
                        df_temporal[mean_col] = df_temporal[col].rolling(window, min_periods=1).mean()

                    # Rolling std (variability)
                    std_col = f"{col}_std_{window}d"
                    if id_col in df_temporal.columns:
                        df_temporal[std_col] = df_temporal.groupby(id_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).std()
                        )
                    else:
                        df_temporal[std_col] = df_temporal[col].rolling(window, min_periods=1).std()

                    # Rolling max (extreme events)
                    max_col = f"{col}_max_{window}d"
                    if id_col in df_temporal.columns:
                        df_temporal[max_col] = df_temporal.groupby(id_col)[col].transform(
                            lambda x: x.rolling(window, min_periods=1).max()
                        )
                    else:
                        df_temporal[max_col] = df_temporal[col].rolling(window, min_periods=1).max()

                    logger.info(f"  Created rolling features for {window}d window: mean, std, max")

        logger.info(f"✅ Created {len(df_temporal.columns) - len(df.columns)} temporal features")

        return df_temporal

    def integrate_climate_with_health(self,
                                     health_df: pd.DataFrame,
                                     coordinate_cols: Tuple[str, str] = ('latitude', 'longitude'),
                                     date_col: str = 'primary_date',
                                     create_features: bool = True) -> pd.DataFrame:
        """Integrate climate data with health dataset using fixed timestamp handling."""

        logger.info("🔗 INTEGRATING CLIMATE DATA WITH HEALTH RECORDS")
        logger.info("="*50)

        # Load climate datasets if not already loaded
        if not self.climate_datasets:
            self.load_climate_datasets()

        # Prepare health data
        df_integrated = health_df.copy()
        lat_col, lon_col = coordinate_cols

        # Filter to records with valid coordinates and dates
        valid_mask = (
            df_integrated[lat_col].notna() &
            df_integrated[lon_col].notna() &
            df_integrated[date_col].notna()
        )

        valid_indices = df_integrated[valid_mask].index
        logger.info(f"📊 Valid records for climate integration: {len(valid_indices):,}/{len(health_df):,}")

        if len(valid_indices) == 0:
            logger.error("❌ No valid coordinate-date records found")
            return health_df

        # Convert dates to pandas timestamps
        df_integrated[date_col] = pd.to_datetime(df_integrated[date_col], errors='coerce')

        # Initialize climate columns
        climate_columns_added = []

        # Process each dataset
        for dataset_name, dataset in self.climate_datasets.items():
            logger.info(f"\n  Processing {dataset_name}...")

            successful_extractions = 0
            failed_extractions = 0

            # Get variables to extract
            vars_to_extract = self.available_variables.get(dataset_name, [])

            # Process in batches for efficiency
            batch_size = 100
            for batch_start in range(0, len(valid_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_indices))
                batch_indices = valid_indices[batch_start:batch_end]

                for idx in batch_indices:
                    row = df_integrated.loc[idx]
                    lat = row[lat_col]
                    lon = row[lon_col]
                    date = row[date_col]

                    # Skip if date is invalid
                    if pd.isna(date):
                        continue

                    # Extract climate data with fixed timestamp handling
                    climate_data = self.extract_point_data_fixed(
                        dataset, lat, lon, date,
                        tolerance_days=1,
                        variables=vars_to_extract
                    )

                    if climate_data:
                        successful_extractions += 1

                        # Add climate variables to dataframe
                        for var_name, value in climate_data.items():
                            col_name = f"{dataset_name}_{var_name}"

                            # Initialize column if needed
                            if col_name not in df_integrated.columns:
                                df_integrated[col_name] = np.nan
                                climate_columns_added.append(col_name)

                            df_integrated.loc[idx, col_name] = value
                    else:
                        failed_extractions += 1

                # Log progress
                if (batch_end % 500) == 0:
                    logger.info(f"    Processed {batch_end}/{len(valid_indices)} records...")

            # Report extraction success rate
            total_attempts = successful_extractions + failed_extractions
            if total_attempts > 0:
                success_rate = (successful_extractions / total_attempts) * 100
                logger.info(f"    ✓ Successful extractions: {successful_extractions}/{total_attempts} ({success_rate:.1f}%)")

        # Create temporal features if requested
        if create_features and climate_columns_added:
            logger.info("\n  Creating temporal features...")

            # Select key temperature variables for temporal features
            temp_vars = [col for col in climate_columns_added
                        if any(t in col.lower() for t in ['tas', 'lst', 'temp'])]

            if temp_vars:
                # Create features for first 2 temperature variables only (to avoid too many features)
                df_integrated = self.create_temporal_features(
                    df_integrated,
                    temp_vars[:2],
                    lag_days=[1, 3, 7],
                    rolling_windows=[3, 7]
                )

        # Generate integration report
        self.integration_report = {
            'total_records': len(health_df),
            'valid_records': len(valid_indices),
            'climate_datasets_used': list(self.climate_datasets.keys()),
            'climate_columns_added': len(climate_columns_added),
            'integration_success_rate': (len(valid_indices) / len(health_df)) * 100,
            'columns_added': climate_columns_added
        }

        logger.info("\n" + "="*50)
        logger.info("✅ CLIMATE INTEGRATION COMPLETE")
        logger.info(f"  • Climate columns added: {len(climate_columns_added)}")
        logger.info(f"  • Total columns now: {len(df_integrated.columns)}")
        logger.info(f"  • Integration success rate: {self.integration_report['integration_success_rate']:.1f}%")

        return df_integrated

    def create_integration_summary(self, df_integrated: pd.DataFrame) -> pd.DataFrame:
        """Create summary of integrated climate data."""

        # Find climate columns
        climate_cols = [col for col in df_integrated.columns
                       if any(dataset in col for dataset in self.climate_datasets.keys())]

        summary_data = []
        for col in climate_cols:
            values = pd.to_numeric(df_integrated[col], errors='coerce')
            non_null = values.notna().sum()

            if non_null > 0:
                summary_data.append({
                    'Variable': col,
                    'Non_Null_Count': non_null,
                    'Completion_%': (non_null / len(df_integrated)) * 100,
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Q25': values.quantile(0.25),
                    'Q50': values.quantile(0.50),
                    'Q75': values.quantile(0.75)
                })

        summary_df = pd.DataFrame(summary_data)

        if not summary_df.empty:
            summary_df = summary_df.sort_values('Completion_%', ascending=False)
            summary_df = summary_df.round(2)

        return summary_df

def test_enhanced_integration():
    """Test the enhanced climate integration with fixed timestamp handling."""

    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING ENHANCED CLIMATE INTEGRATION")
    logger.info("="*70)

    # Initialize integrator
    integrator = EnhancedClimateIntegrator()

    # Load sample health data
    logger.info("\n📊 Loading sample health data...")
    health_data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/MASTER_INTEGRATED_DATASET.csv"

    # Load a subset for testing (first 500 records with valid coordinates)
    df_full = pd.read_csv(health_data_path)

    # Check columns and adapt
    logger.info(f"  Available columns: {list(df_full.columns[:10])}...")

    # Find coordinate and date columns
    lat_col = 'latitude' if 'latitude' in df_full.columns else 'lat' if 'lat' in df_full.columns else None
    lon_col = 'longitude' if 'longitude' in df_full.columns else 'lon' if 'lon' in df_full.columns else None
    date_col = 'primary_date' if 'primary_date' in df_full.columns else 'date' if 'date' in df_full.columns else 'survey_year'
    id_col = 'patient_id' if 'patient_id' in df_full.columns else 'gcro_id' if 'gcro_id' in df_full.columns else df_full.columns[0]

    # For GCRO data, we need to create dates from survey year
    if date_col == 'survey_year' and 'survey_year' in df_full.columns:
        df_full['primary_date'] = pd.to_datetime(df_full['survey_year'].astype(str) + '-07-01')
        date_col = 'primary_date'

    # Filter to records with coordinates
    if lat_col and lon_col:
        df_with_coords = df_full[
            df_full[lat_col].notna() &
            df_full[lon_col].notna()
        ].head(500)
    else:
        logger.error("  ❌ No coordinate columns found!")
        df_with_coords = df_full.head(500)

    logger.info(f"  Loaded {len(df_with_coords):,} sample records with valid coordinates")
    logger.info(f"  Using columns: ID={id_col}, Lat={lat_col}, Lon={lon_col}, Date={date_col}")

    # Show sample of data if we have the columns
    if lat_col and lon_col and date_col:
        logger.info("\n  Sample records:")
        sample = df_with_coords[[id_col, lat_col, lon_col, date_col]].head(3)
        for _, row in sample.iterrows():
            logger.info(f"    ID: {row[id_col]}, Lat: {row[lat_col]:.4f}, "
                       f"Lon: {row[lon_col]:.4f}, Date: {str(row[date_col])}")

    # Test climate integration
    df_integrated = integrator.integrate_climate_with_health(
        df_with_coords,
        coordinate_cols=(lat_col, lon_col) if lat_col and lon_col else ('longitude', 'latitude'),
        date_col=date_col if date_col else 'primary_date',
        create_features=True
    )

    # Create summary
    summary = integrator.create_integration_summary(df_integrated)

    if not summary.empty:
        logger.info("\n📋 Climate Integration Summary:")
        logger.info("\n" + summary.to_string(index=False))

    # Save test results
    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)

    # Save integrated data
    test_output_path = output_dir / "climate_integration_enhanced_test.csv"
    df_integrated.to_csv(test_output_path, index=False)

    # Save summary
    summary_output_path = output_dir / "climate_integration_summary.csv"
    summary.to_csv(summary_output_path, index=False)

    # Save integration report
    report_path = output_dir / "climate_integration_report.json"
    with open(report_path, 'w') as f:
        json.dump(integrator.integration_report, f, indent=2)

    logger.info(f"\n✅ Test complete. Results saved to:")
    logger.info(f"  • Integrated data: {test_output_path}")
    logger.info(f"  • Summary: {summary_output_path}")
    logger.info(f"  • Report: {report_path}")

    return df_integrated, summary

def main():
    """Main execution function."""
    df_integrated, summary = test_enhanced_integration()
    return df_integrated, summary

if __name__ == "__main__":
    df_integrated, summary = main()