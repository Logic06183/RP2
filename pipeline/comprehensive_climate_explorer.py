#!/usr/bin/env python3
"""
Comprehensive Climate Data Explorer for HEAT Research
======================================================

This script explores ALL available climate variables in the zarr files,
documenting their temporal resolution, coverage, and suitability for
heat-health analysis.

Author: HEAT Research Team
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import xarray as xr
import zarr
import logging
from pathlib import Path
from datetime import datetime
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveClimateExplorer:
    """Explores and documents all climate variables in zarr datasets."""

    def __init__(self, climate_dir: str = "/home/cparker/selected_data_all/data/RP2_subsets/JHB"):
        """Initialize the climate explorer."""
        self.climate_dir = Path(climate_dir)
        self.datasets_info = {}
        self.all_variables = {}
        self.variable_metadata = {}

    def explore_all_datasets(self):
        """Explore all zarr datasets and document variables."""
        logger.info("🌍 COMPREHENSIVE CLIMATE DATA EXPLORATION")
        logger.info("=" * 60)

        # Find all zarr files
        zarr_files = list(self.climate_dir.glob("*.zarr"))
        logger.info(f"Found {len(zarr_files)} zarr datasets to explore")

        for zarr_path in sorted(zarr_files):
            dataset_name = zarr_path.stem
            logger.info(f"\n📊 Exploring: {dataset_name}")
            logger.info("-" * 40)

            try:
                # Open dataset
                ds = xr.open_zarr(zarr_path)

                # Store dataset info
                dataset_info = {
                    'name': dataset_name,
                    'path': str(zarr_path),
                    'dimensions': dict(ds.dims),
                    'coordinates': list(ds.coords.keys()),
                    'variables': {},
                    'temporal_range': None,
                    'spatial_extent': None,
                    'total_size_gb': None
                }

                # Analyze dimensions
                logger.info(f"  Dimensions: {dict(ds.dims)}")

                # Analyze temporal coverage if time dimension exists
                if 'time' in ds.dims:
                    time_array = ds.time.values
                    time_start = pd.to_datetime(str(time_array[0]))
                    time_end = pd.to_datetime(str(time_array[-1]))
                    time_steps = len(time_array)

                    # Calculate temporal resolution
                    if time_steps > 1:
                        time_diff = pd.to_datetime(str(time_array[1])) - pd.to_datetime(str(time_array[0]))
                        if time_diff.days >= 365:
                            temporal_res = "yearly"
                        elif time_diff.days >= 28:
                            temporal_res = "monthly"
                        elif time_diff.days >= 7:
                            temporal_res = "weekly"
                        elif time_diff.days >= 1:
                            temporal_res = "daily"
                        elif time_diff.seconds >= 3600:
                            temporal_res = f"{time_diff.seconds // 3600} hourly"
                        else:
                            temporal_res = "sub-hourly"
                    else:
                        temporal_res = "single time point"

                    dataset_info['temporal_range'] = {
                        'start': str(time_start),
                        'end': str(time_end),
                        'steps': time_steps,
                        'resolution': temporal_res,
                        'duration_years': (time_end - time_start).days / 365.25
                    }

                    logger.info(f"  Temporal: {time_start.date()} to {time_end.date()} ({time_steps} steps, {temporal_res})")

                # Analyze spatial coverage
                lat_names = ['latitude', 'lat', 'y']
                lon_names = ['longitude', 'lon', 'x']

                lat_coord = None
                lon_coord = None

                for coord in ds.coords:
                    if any(lat_name in coord.lower() for lat_name in lat_names):
                        lat_coord = coord
                    if any(lon_name in coord.lower() for lon_name in lon_names):
                        lon_coord = coord

                if lat_coord and lon_coord:
                    lat_values = ds[lat_coord].values
                    lon_values = ds[lon_coord].values

                    dataset_info['spatial_extent'] = {
                        'lat_min': float(np.min(lat_values)),
                        'lat_max': float(np.max(lat_values)),
                        'lon_min': float(np.min(lon_values)),
                        'lon_max': float(np.max(lon_values)),
                        'lat_points': len(lat_values),
                        'lon_points': len(lon_values),
                        'total_grid_points': len(lat_values) * len(lon_values)
                    }

                    logger.info(f"  Spatial: Lat [{np.min(lat_values):.3f}, {np.max(lat_values):.3f}] "
                              f"Lon [{np.min(lon_values):.3f}, {np.max(lon_values):.3f}]")
                    logger.info(f"  Grid: {len(lat_values)} x {len(lon_values)} = "
                              f"{len(lat_values) * len(lon_values):,} points")

                # Analyze each variable
                logger.info(f"  Variables ({len(ds.data_vars)}):")
                for var_name in ds.data_vars:
                    var = ds[var_name]

                    # Get variable metadata
                    var_info = {
                        'name': var_name,
                        'dataset': dataset_name,
                        'dims': list(var.dims),
                        'shape': list(var.shape),
                        'dtype': str(var.dtype),
                        'units': var.attrs.get('units', 'unknown'),
                        'long_name': var.attrs.get('long_name', var.attrs.get('standard_name', var_name)),
                        'attributes': dict(var.attrs)
                    }

                    # Try to get statistics (sample if too large)
                    try:
                        # Sample data for statistics if dataset is large
                        if var.size > 1e7:  # If more than 10 million values
                            # Sample every 10th time step and every 5th spatial point
                            sample_slice = {}
                            for dim in var.dims:
                                if 'time' in dim:
                                    sample_slice[dim] = slice(None, None, 10)
                                elif any(coord in dim for coord in ['lat', 'lon', 'x', 'y']):
                                    sample_slice[dim] = slice(None, None, 5)
                            var_sample = var.isel(sample_slice)
                            logger.info(f"    Sampling large variable for statistics...")
                        else:
                            var_sample = var

                        # Compute statistics
                        var_data = var_sample.values.flatten()
                        var_data = var_data[~np.isnan(var_data)]  # Remove NaNs

                        if len(var_data) > 0:
                            var_info['statistics'] = {
                                'mean': float(np.mean(var_data)),
                                'std': float(np.std(var_data)),
                                'min': float(np.min(var_data)),
                                'max': float(np.max(var_data)),
                                'median': float(np.median(var_data)),
                                'q25': float(np.percentile(var_data, 25)),
                                'q75': float(np.percentile(var_data, 75)),
                                'non_null_count': int(len(var_data)),
                                'null_count': int(var.size - len(var_data)),
                                'completeness': float(len(var_data) / var.size * 100)
                            }
                        else:
                            var_info['statistics'] = {'completeness': 0.0}

                    except Exception as e:
                        logger.warning(f"    Could not compute statistics for {var_name}: {e}")
                        var_info['statistics'] = None

                    # Store variable info
                    dataset_info['variables'][var_name] = var_info

                    # Add to all variables catalog
                    full_var_name = f"{dataset_name}_{var_name}"
                    self.all_variables[full_var_name] = var_info

                    # Log variable summary
                    units = var_info.get('units', 'unknown')
                    if var_info.get('statistics'):
                        stats = var_info['statistics']
                        logger.info(f"    • {var_name}: {var_info['long_name']} [{units}]")
                        if stats.get('mean') is not None:
                            logger.info(f"      Range: [{stats['min']:.2f}, {stats['max']:.2f}], "
                                      f"Mean: {stats['mean']:.2f}, "
                                      f"Completeness: {stats['completeness']:.1f}%")
                    else:
                        logger.info(f"    • {var_name}: {var_info['long_name']} [{units}]")

                # Calculate approximate dataset size
                try:
                    total_size = sum(var.nbytes for var in ds.data_vars.values())
                    dataset_info['total_size_gb'] = total_size / (1024**3)
                    logger.info(f"  Dataset size: ~{dataset_info['total_size_gb']:.2f} GB")
                except:
                    pass

                # Store dataset info
                self.datasets_info[dataset_name] = dataset_info

                # Close dataset
                ds.close()

            except Exception as e:
                logger.error(f"  ❌ Failed to explore {dataset_name}: {e}")
                continue

        logger.info(f"\n✅ Exploration complete: {len(self.datasets_info)} datasets, "
                   f"{len(self.all_variables)} total variables")

        return self.datasets_info

    def identify_heat_health_variables(self):
        """Identify optimal variables for heat-health analysis based on scientific criteria."""
        logger.info("\n🔬 IDENTIFYING OPTIMAL HEAT-HEALTH VARIABLES")
        logger.info("=" * 60)

        # Define priority variables for heat-health analysis
        heat_health_priorities = {
            'temperature': {
                'keywords': ['temperature', 'temp', 'tas', 't2m', 'tmax', 'tmin'],
                'importance': 'critical',
                'reason': 'Direct heat exposure metric'
            },
            'humidity': {
                'keywords': ['humidity', 'rh', 'dewpoint', 'dew', 'specific_humidity', 'q2m', 'vapor'],
                'importance': 'critical',
                'reason': 'Affects heat stress and physiological cooling'
            },
            'heat_index': {
                'keywords': ['heat_index', 'apparent_temperature', 'feels_like', 'wbgt', 'utci'],
                'importance': 'high',
                'reason': 'Composite heat stress metric'
            },
            'solar_radiation': {
                'keywords': ['radiation', 'solar', 'shortwave', 'ssrd', 'ssr', 'uv'],
                'importance': 'high',
                'reason': 'Direct heat load on humans'
            },
            'wind': {
                'keywords': ['wind', 'u10', 'v10', 'wind_speed', 'ws'],
                'importance': 'moderate',
                'reason': 'Affects evaporative cooling'
            },
            'pressure': {
                'keywords': ['pressure', 'msl', 'sp', 'surface_pressure'],
                'importance': 'moderate',
                'reason': 'Affects oxygen availability and circulation'
            },
            'precipitation': {
                'keywords': ['precipitation', 'rain', 'prcp', 'precip', 'tp'],
                'importance': 'moderate',
                'reason': 'Cooling effect and humidity changes'
            },
            'land_surface_temp': {
                'keywords': ['lst', 'land_surface', 'skin_temperature'],
                'importance': 'moderate',
                'reason': 'Urban heat island effects'
            }
        }

        # Categorize available variables
        categorized_variables = {category: [] for category in heat_health_priorities}
        uncategorized = []

        for var_full_name, var_info in self.all_variables.items():
            categorized = False
            var_name_lower = var_info['name'].lower()
            long_name_lower = var_info.get('long_name', '').lower()

            for category, config in heat_health_priorities.items():
                if any(keyword in var_name_lower or keyword in long_name_lower
                      for keyword in config['keywords']):
                    categorized_variables[category].append({
                        'full_name': var_full_name,
                        'dataset': var_info['dataset'],
                        'variable': var_info['name'],
                        'units': var_info.get('units', 'unknown'),
                        'completeness': var_info.get('statistics', {}).get('completeness', 0)
                    })
                    categorized = True
                    break

            if not categorized:
                uncategorized.append(var_full_name)

        # Log categorized variables
        logger.info("\n📋 Categorized Variables for Heat-Health Analysis:")

        selected_variables = []

        for category, config in heat_health_priorities.items():
            vars_in_category = categorized_variables[category]
            if vars_in_category:
                logger.info(f"\n{category.upper()} (Importance: {config['importance']})")
                logger.info(f"  Reason: {config['reason']}")
                logger.info(f"  Available variables ({len(vars_in_category)}):")

                # Sort by completeness
                vars_in_category.sort(key=lambda x: x['completeness'], reverse=True)

                for var in vars_in_category[:5]:  # Show top 5
                    logger.info(f"    • {var['variable']} from {var['dataset']}")
                    logger.info(f"      Units: {var['units']}, Completeness: {var['completeness']:.1f}%")

                    # Select best variable from critical/high importance categories
                    if config['importance'] in ['critical', 'high'] and var['completeness'] > 50:
                        selected_variables.append(var)

        if uncategorized:
            logger.info(f"\n📦 Other variables ({len(uncategorized)} uncategorized)")
            for var in uncategorized[:10]:  # Show first 10
                logger.info(f"  • {var}")

        self.selected_variables = selected_variables

        logger.info(f"\n✅ Selected {len(selected_variables)} optimal variables for analysis")

        return selected_variables

    def create_variable_summary_table(self):
        """Create comprehensive summary tables of all climate variables."""
        logger.info("\n📊 Creating Variable Summary Tables")

        # Create main summary dataframe
        summary_data = []

        for dataset_name, dataset_info in self.datasets_info.items():
            for var_name, var_info in dataset_info['variables'].items():
                row = {
                    'Dataset': dataset_name,
                    'Variable': var_name,
                    'Long_Name': var_info.get('long_name', var_name),
                    'Units': var_info.get('units', 'unknown'),
                    'Dimensions': ', '.join(var_info.get('dims', [])),
                    'Shape': str(var_info.get('shape', [])),
                }

                # Add temporal info
                if dataset_info.get('temporal_range'):
                    temporal = dataset_info['temporal_range']
                    row['Time_Start'] = temporal['start'][:10]  # Date only
                    row['Time_End'] = temporal['end'][:10]
                    row['Time_Steps'] = temporal['steps']
                    row['Time_Resolution'] = temporal['resolution']
                else:
                    row['Time_Start'] = 'N/A'
                    row['Time_End'] = 'N/A'
                    row['Time_Steps'] = 'N/A'
                    row['Time_Resolution'] = 'N/A'

                # Add statistics
                if var_info.get('statistics'):
                    stats = var_info['statistics']
                    row['Mean'] = f"{stats.get('mean', np.nan):.2f}" if stats.get('mean') is not None else 'N/A'
                    row['Std'] = f"{stats.get('std', np.nan):.2f}" if stats.get('std') is not None else 'N/A'
                    row['Min'] = f"{stats.get('min', np.nan):.2f}" if stats.get('min') is not None else 'N/A'
                    row['Max'] = f"{stats.get('max', np.nan):.2f}" if stats.get('max') is not None else 'N/A'
                    row['Completeness_%'] = f"{stats.get('completeness', 0):.1f}"
                else:
                    row['Mean'] = 'N/A'
                    row['Std'] = 'N/A'
                    row['Min'] = 'N/A'
                    row['Max'] = 'N/A'
                    row['Completeness_%'] = '0.0'

                summary_data.append(row)

        # Create dataframe
        summary_df = pd.DataFrame(summary_data)

        # Sort by dataset and completeness
        summary_df['Completeness_numeric'] = summary_df['Completeness_%'].str.rstrip('%').astype(float)
        summary_df = summary_df.sort_values(['Dataset', 'Completeness_numeric'], ascending=[True, False])
        summary_df = summary_df.drop('Completeness_numeric', axis=1)

        # Save to CSV
        output_path = Path("../data/climate_variables_comprehensive_summary.csv")
        summary_df.to_csv(output_path, index=False)
        logger.info(f"  Saved comprehensive summary to: {output_path}")

        # Create dataset-level summary
        dataset_summary = []
        for dataset_name, dataset_info in self.datasets_info.items():
            row = {
                'Dataset': dataset_name,
                'Variables': len(dataset_info['variables']),
                'Size_GB': f"{dataset_info.get('total_size_gb', 0):.2f}" if dataset_info.get('total_size_gb') else 'N/A'
            }

            # Add temporal info
            if dataset_info.get('temporal_range'):
                temporal = dataset_info['temporal_range']
                row['Time_Coverage'] = f"{temporal['start'][:10]} to {temporal['end'][:10]}"
                row['Duration_Years'] = f"{temporal['duration_years']:.1f}"
                row['Resolution'] = temporal['resolution']
            else:
                row['Time_Coverage'] = 'N/A'
                row['Duration_Years'] = 'N/A'
                row['Resolution'] = 'N/A'

            # Add spatial info
            if dataset_info.get('spatial_extent'):
                spatial = dataset_info['spatial_extent']
                row['Lat_Range'] = f"[{spatial['lat_min']:.2f}, {spatial['lat_max']:.2f}]"
                row['Lon_Range'] = f"[{spatial['lon_min']:.2f}, {spatial['lon_max']:.2f}]"
                row['Grid_Points'] = f"{spatial['total_grid_points']:,}"
            else:
                row['Lat_Range'] = 'N/A'
                row['Lon_Range'] = 'N/A'
                row['Grid_Points'] = 'N/A'

            dataset_summary.append(row)

        dataset_summary_df = pd.DataFrame(dataset_summary)

        # Save dataset summary
        dataset_output_path = Path("../data/climate_datasets_summary.csv")
        dataset_summary_df.to_csv(dataset_output_path, index=False)
        logger.info(f"  Saved dataset summary to: {dataset_output_path}")

        return summary_df, dataset_summary_df

    def save_metadata_json(self):
        """Save complete metadata to JSON for future reference."""
        output_path = Path("../data/climate_metadata_complete.json")

        # Prepare metadata for JSON serialization
        metadata = {
            'exploration_date': datetime.now().isoformat(),
            'datasets': self.datasets_info,
            'total_datasets': len(self.datasets_info),
            'total_variables': len(self.all_variables),
            'selected_heat_health_variables': self.selected_variables if hasattr(self, 'selected_variables') else []
        }

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        metadata = convert_types(metadata)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Saved complete metadata to: {output_path}")

        return output_path

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("COMPREHENSIVE CLIMATE DATA EXPLORATION FOR HEAT RESEARCH")
    print("="*70)

    # Initialize explorer
    explorer = ComprehensiveClimateExplorer()

    # Explore all datasets
    datasets_info = explorer.explore_all_datasets()

    # Identify optimal variables for heat-health analysis
    selected_variables = explorer.identify_heat_health_variables()

    # Create summary tables
    summary_df, dataset_summary_df = explorer.create_variable_summary_table()

    # Save metadata
    metadata_path = explorer.save_metadata_json()

    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"  • Datasets explored: {len(datasets_info)}")
    print(f"  • Total variables found: {len(explorer.all_variables)}")
    print(f"  • Variables selected for heat-health: {len(selected_variables)}")
    print(f"\n📁 Output Files:")
    print(f"  • Climate variables summary: ../data/climate_variables_comprehensive_summary.csv")
    print(f"  • Dataset summary: ../data/climate_datasets_summary.csv")
    print(f"  • Complete metadata: ../data/climate_metadata_complete.json")

    return explorer

if __name__ == "__main__":
    explorer = main()