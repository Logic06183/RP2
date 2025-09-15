#!/usr/bin/env python3
"""
HEAT Climate Data Integration Tool
=================================

Properly integrates climate data from zarr files with health data based on coordinates and time.

Author: Craig Parker
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ClimateDataIntegrator:
    """Integrates climate data from zarr files with health data."""

    def __init__(self, climate_data_dir: str = "../data/climate/johannesburg"):
        """Initialize the climate data integrator."""
        self.climate_data_dir = Path(climate_data_dir)
        self.climate_datasets = {}
        self.integration_report = {}

    def discover_climate_files(self) -> Dict[str, Path]:
        """Discover available climate data files."""
        logger.info("🔍 Discovering climate data files...")

        climate_files = {}
        for file_path in self.climate_data_dir.glob("*.zarr"):
            if file_path.exists() and file_path.is_dir():
                climate_files[file_path.stem] = file_path
                logger.info(f"   Found: {file_path.stem}")

        logger.info(f"✅ Discovered {len(climate_files)} climate datasets")
        return climate_files

    def load_climate_dataset(self, file_path: Path, dataset_name: str) -> Optional[xr.Dataset]:
        """Load a single climate dataset from zarr."""
        try:
            logger.info(f"📊 Loading {dataset_name}...")
            ds = xr.open_zarr(file_path)

            # Basic dataset info
            logger.info(f"   Variables: {list(ds.data_vars.keys())}")
            logger.info(f"   Dimensions: {dict(ds.dims)}")

            # Time range
            if 'time' in ds.dims:
                time_range = f"{ds.time.min().values} to {ds.time.max().values}"
                logger.info(f"   Time range: {time_range}")

            # Spatial range
            if 'latitude' in ds.dims and 'longitude' in ds.dims:
                lat_range = f"{ds.latitude.min().values:.3f} to {ds.latitude.max().values:.3f}"
                lon_range = f"{ds.longitude.min().values:.3f} to {ds.longitude.max().values:.3f}"
                logger.info(f"   Spatial: Lat {lat_range}, Lon {lon_range}")

            return ds

        except Exception as e:
            logger.error(f"   ❌ Failed to load {dataset_name}: {e}")
            return None

    def load_all_climate_data(self) -> Dict[str, xr.Dataset]:
        """Load all available climate datasets."""
        logger.info("🌡️  Loading all climate datasets...")

        climate_files = self.discover_climate_files()

        for name, file_path in climate_files.items():
            dataset = self.load_climate_dataset(file_path, name)
            if dataset is not None:
                self.climate_datasets[name] = dataset

        logger.info(f"✅ Loaded {len(self.climate_datasets)} climate datasets")
        return self.climate_datasets

    def extract_point_data(self,
                          dataset: xr.Dataset,
                          lat: float,
                          lon: float,
                          date: pd.Timestamp,
                          tolerance_days: int = 1) -> Dict[str, float]:
        """Extract climate data for a specific point and time."""

        try:
            # Find nearest spatial point
            if 'latitude' in dataset.dims and 'longitude' in dataset.dims:
                point_data = dataset.sel(
                    latitude=lat,
                    longitude=lon,
                    method='nearest'
                )
            else:
                # Try alternative coordinate names
                coord_names = list(dataset.dims.keys())
                lat_coord = next((name for name in coord_names if 'lat' in name.lower()), None)
                lon_coord = next((name for name in coord_names if 'lon' in name.lower()), None)

                if lat_coord and lon_coord:
                    point_data = dataset.sel(
                        {lat_coord: lat, lon_coord: lon},
                        method='nearest'
                    )
                else:
                    logger.warning(f"   ⚠️  No spatial coordinates found in dataset")
                    return {}

            # Find nearest time point
            if 'time' in point_data.dims:
                # Convert date to same format as dataset time
                target_time = pd.to_datetime(date)

                # Find nearest time within tolerance
                time_diff = np.abs(point_data.time - target_time)
                min_diff = time_diff.min()

                if min_diff <= pd.Timedelta(days=tolerance_days):
                    nearest_time_data = point_data.sel(time=target_time, method='nearest')
                else:
                    logger.warning(f"   ⚠️  No time data within {tolerance_days} days of {date}")
                    return {}
            else:
                # Use all data if no time dimension
                nearest_time_data = point_data

            # Extract values for all variables
            extracted_data = {}
            for var_name in nearest_time_data.data_vars:
                try:
                    value = float(nearest_time_data[var_name].values)
                    if not np.isnan(value):
                        extracted_data[var_name] = value
                except Exception as e:
                    logger.debug(f"   Could not extract {var_name}: {e}")

            return extracted_data

        except Exception as e:
            logger.warning(f"   ⚠️  Point extraction failed: {e}")
            return {}

    def integrate_climate_with_health_data(self,
                                         health_df: pd.DataFrame,
                                         coordinate_cols: Tuple[str, str] = ('latitude', 'longitude'),
                                         date_col: str = 'primary_date',
                                         climate_datasets: Optional[List[str]] = None) -> pd.DataFrame:
        """Integrate climate data with health dataset."""

        logger.info("🔗 Integrating climate data with health data...")

        # Load climate data if not already loaded
        if not self.climate_datasets:
            self.load_all_climate_data()

        # Select datasets to use
        if climate_datasets is None:
            # Use ERA5 and SAAQIS by default (most reliable)
            climate_datasets = ['ERA5_tas_native', 'ERA5-Land_tas_native', 'SAAQIS_with_climate_variables']

        # Filter to available datasets
        available_datasets = [name for name in climate_datasets if name in self.climate_datasets]
        logger.info(f"   Using datasets: {available_datasets}")

        if not available_datasets:
            logger.error("   ❌ No climate datasets available")
            return health_df

        # Prepare health data
        df_with_climate = health_df.copy()
        lat_col, lon_col = coordinate_cols

        # Filter to records with valid coordinates and dates
        valid_coords = (
            df_with_climate[lat_col].notna() &
            df_with_climate[lon_col].notna() &
            df_with_climate[date_col].notna()
        )

        valid_df = df_with_climate[valid_coords].copy()
        logger.info(f"   Valid coordinate-date records: {len(valid_df):,}")

        if len(valid_df) == 0:
            logger.error("   ❌ No valid coordinate-date records found")
            return health_df

        # Convert dates
        valid_df[date_col] = pd.to_datetime(valid_df[date_col], errors='coerce')
        valid_df = valid_df[valid_df[date_col].notna()]

        logger.info(f"   Records after date parsing: {len(valid_df):,}")

        # Initialize climate columns
        climate_columns = []

        # Extract climate data for each record
        for dataset_name in available_datasets:
            dataset = self.climate_datasets[dataset_name]
            logger.info(f"   Processing {dataset_name}...")

            # Track successful extractions
            successful_extractions = 0

            for idx, row in valid_df.iterrows():
                lat = row[lat_col]
                lon = row[lon_col]
                date = row[date_col]

                # Extract climate data
                climate_data = self.extract_point_data(dataset, lat, lon, date)

                if climate_data:
                    successful_extractions += 1

                    # Add climate variables to dataframe
                    for var_name, value in climate_data.items():
                        col_name = f"{dataset_name}_{var_name}"
                        if col_name not in climate_columns:
                            climate_columns.append(col_name)
                            df_with_climate[col_name] = np.nan

                        df_with_climate.loc[idx, col_name] = value

            extraction_rate = (successful_extractions / len(valid_df)) * 100
            logger.info(f"     Successful extractions: {successful_extractions}/{len(valid_df)} ({extraction_rate:.1f}%)")

        # Generate integration report
        self.integration_report = {
            'total_records': len(health_df),
            'valid_coordinate_records': len(valid_df),
            'climate_datasets_used': available_datasets,
            'climate_columns_added': len(climate_columns),
            'integration_success_rate': (len(valid_df) / len(health_df)) * 100 if len(health_df) > 0 else 0
        }

        logger.info(f"✅ Climate integration complete:")
        logger.info(f"   Climate columns added: {len(climate_columns)}")
        logger.info(f"   Integration success rate: {self.integration_report['integration_success_rate']:.1f}%")

        return df_with_climate

    def create_climate_summary(self, df_with_climate: pd.DataFrame) -> pd.DataFrame:
        """Create summary of integrated climate data."""

        # Find climate columns
        climate_cols = [col for col in df_with_climate.columns
                       if any(dataset in col for dataset in self.climate_datasets.keys())]

        if not climate_cols:
            logger.warning("No climate columns found for summary")
            return pd.DataFrame()

        summary_data = []
        for col in climate_cols:
            values = pd.to_numeric(df_with_climate[col], errors='coerce')
            non_null = values.notna().sum()
            completion_rate = (non_null / len(df_with_climate)) * 100

            if non_null > 0:
                summary_data.append({
                    'Variable': col,
                    'Count': non_null,
                    'Completion_Rate_%': f'{completion_rate:.1f}%',
                    'Mean': f'{values.mean():.2f}',
                    'Std': f'{values.std():.2f}',
                    'Min': f'{values.min():.2f}',
                    'Max': f'{values.max():.2f}'
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Count', ascending=False)

        return summary_df

def test_climate_integration():
    """Test the climate data integration."""

    print("🧪 Testing Climate Data Integration")
    print("=" * 50)

    # Initialize integrator
    integrator = ClimateDataIntegrator()

    # Load sample health data
    print("📊 Loading sample health data...")
    health_data_path = "../data/MASTER_INTEGRATED_DATASET.csv"

    # Load just a subset for testing
    df = pd.read_csv(health_data_path, nrows=1000)
    print(f"   Loaded {len(df):,} sample records")

    # Test climate integration
    df_integrated = integrator.integrate_climate_with_health_data(df)

    # Create summary
    summary = integrator.create_climate_summary(df_integrated)

    print(f"\n📋 Integration Summary:")
    print(summary.to_string(index=False))

    # Save test results
    output_dir = Path("../data")
    test_output_path = output_dir / "climate_integration_test.csv"
    df_integrated.to_csv(test_output_path, index=False)

    summary_output_path = output_dir / "climate_summary_test.csv"
    summary.to_csv(summary_output_path, index=False)

    print(f"\n✅ Test complete. Results saved to:")
    print(f"   {test_output_path}")
    print(f"   {summary_output_path}")

    return df_integrated, summary

def main():
    """Main execution function."""

    # Test climate integration
    df_integrated, summary = test_climate_integration()

    return df_integrated, summary

if __name__ == "__main__":
    df_integrated, summary = main()