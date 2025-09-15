#!/usr/bin/env python3
"""
Quick Climate Data Explorer for HEAT Research
==============================================

Efficiently explores climate zarr files without loading full datasets.

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def explore_climate_data():
    """Quick exploration of all climate datasets."""

    climate_dir = Path("/home/cparker/selected_data_all/data/RP2_subsets/JHB")
    zarr_files = list(climate_dir.glob("*.zarr"))

    logger.info(f"Found {len(zarr_files)} climate datasets")
    logger.info("="*60)

    all_datasets = {}
    all_variables = []

    for zarr_path in sorted(zarr_files):
        dataset_name = zarr_path.stem
        logger.info(f"\n{dataset_name}")
        logger.info("-"*40)

        try:
            # Open with zarr directly for metadata
            z = zarr.open(zarr_path, mode='r')

            # Get array names (variables)
            variables = [name for name in z.array_keys()]

            # Also open with xarray for coordinate info
            ds = xr.open_zarr(zarr_path, chunks='auto')

            # Get basic info
            info = {
                'name': dataset_name,
                'variables': {},
                'dims': dict(ds.dims),
                'coords': list(ds.coords.keys())
            }

            # Quick time range check
            if 'time' in ds.dims:
                time_vals = ds.time.values
                info['time_range'] = {
                    'start': str(pd.to_datetime(time_vals[0])),
                    'end': str(pd.to_datetime(time_vals[-1])),
                    'steps': len(time_vals)
                }
                logger.info(f"  Time: {info['time_range']['start'][:10]} to {info['time_range']['end'][:10]} ({info['time_range']['steps']} steps)")

            # Check each variable
            for var_name in ds.data_vars:
                var = ds[var_name]
                var_info = {
                    'dataset': dataset_name,
                    'name': var_name,
                    'dims': list(var.dims),
                    'shape': list(var.shape),
                    'units': var.attrs.get('units', 'unknown'),
                    'long_name': var.attrs.get('long_name', var.attrs.get('standard_name', var_name))
                }

                info['variables'][var_name] = var_info
                all_variables.append(var_info)

                logger.info(f"  • {var_name}: {var_info['long_name']} [{var_info['units']}]")

            all_datasets[dataset_name] = info

            ds.close()

        except Exception as e:
            logger.error(f"  Error: {e}")
            continue

    # Categorize variables by type
    logger.info("\n" + "="*60)
    logger.info("VARIABLE CATEGORIES FOR HEAT-HEALTH ANALYSIS")
    logger.info("="*60)

    categories = {
        'Temperature': [],
        'Humidity': [],
        'Wind': [],
        'Radiation': [],
        'Pressure': [],
        'Precipitation': [],
        'Other': []
    }

    for var in all_variables:
        name_lower = var['name'].lower()
        long_name_lower = var.get('long_name', '').lower()

        if any(t in name_lower or t in long_name_lower for t in ['temp', 'tas', 't2m', 'lst']):
            categories['Temperature'].append(f"{var['dataset']}/{var['name']}")
        elif any(h in name_lower or h in long_name_lower for h in ['humid', 'rh', 'dewpoint', 'vapor']):
            categories['Humidity'].append(f"{var['dataset']}/{var['name']}")
        elif any(w in name_lower or w in long_name_lower for w in ['wind', 'ws', 'u10', 'v10']):
            categories['Wind'].append(f"{var['dataset']}/{var['name']}")
        elif any(r in name_lower or r in long_name_lower for r in ['radiation', 'solar', 'shortwave']):
            categories['Radiation'].append(f"{var['dataset']}/{var['name']}")
        elif any(p in name_lower or p in long_name_lower for p in ['pressure', 'msl']):
            categories['Pressure'].append(f"{var['dataset']}/{var['name']}")
        elif any(p in name_lower or p in long_name_lower for p in ['precip', 'rain']):
            categories['Precipitation'].append(f"{var['dataset']}/{var['name']}")
        else:
            categories['Other'].append(f"{var['dataset']}/{var['name']}")

    for category, vars in categories.items():
        if vars:
            logger.info(f"\n{category} ({len(vars)} variables):")
            for v in vars[:5]:  # Show first 5
                logger.info(f"  • {v}")

    # Save summary
    summary = {
        'exploration_date': datetime.now().isoformat(),
        'datasets': all_datasets,
        'total_datasets': len(all_datasets),
        'total_variables': len(all_variables),
        'categories': {k: len(v) for k, v in categories.items()}
    }

    with open('../data/climate_quick_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Summary saved to ../data/climate_quick_summary.json")

    return all_datasets, categories

if __name__ == "__main__":
    all_datasets, categories = explore_climate_data()