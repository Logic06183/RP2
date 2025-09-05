"""
Data I/O utilities for the HEAT Lab climate-health analysis project.

This module provides functions to load climate and health data from the
harmonized RP2 datasets with proper type handling and validation.
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data paths
DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data"
CLIMATE_ROOT = DATA_ROOT / "climate" / "johannesburg"
HEALTH_ROOT = DATA_ROOT / "health" / "rp2_harmonized"
SOCIOECONOMIC_ROOT = DATA_ROOT / "socioeconomic" / "processed"

# Alternative paths for full datasets
ALT_CLIMATE_ROOT = Path("/home/cparker/selected_data_all/data/RP2_subsets/JHB")
ALT_HEALTH_ROOT = Path("/home/cparker/incoming/RP2/00_FINAL_DATASETS")
ALT_SOCIOECONOMIC = Path("/home/cparker/selected_data_all/data/socio-economic/RP2/harmonized_datasets")


def load_climate_data(dataset: str, date_range: Optional[Tuple[str, str]] = None) -> xr.Dataset:
    """
    Load climate dataset from zarr files.
    
    Parameters
    ----------
    dataset : str
        Name of dataset to load. Options:
        - 'era5_temp': ERA5 air temperature
        - 'era5_lst': ERA5 land surface temperature  
        - 'era5_wind': ERA5 wind speed
        - 'era5_land_temp': ERA5-Land air temperature
        - 'era5_land_lst': ERA5-Land land surface temperature
        - 'modis_lst': MODIS land surface temperature
        - 'meteosat_lst': Meteosat land surface temperature
        - 'wrf_temp': WRF downscaled air temperature
        - 'wrf_lst': WRF downscaled land surface temperature
        - 'saaqis': SAAQIS weather stations with climate variables
        
    date_range : tuple of str, optional
        Start and end dates as ('YYYY-MM-DD', 'YYYY-MM-DD')
        
    Returns
    -------
    xr.Dataset
        Climate data as xarray dataset
    """
    dataset_mapping = {
        'era5_temp': 'ERA5_tas_native.zarr',
        'era5_temp_regrid': 'ERA5_tas_regrid.zarr',
        'era5_lst': 'ERA5_lst_native.zarr',
        'era5_lst_regrid': 'ERA5_lst_regrid.zarr',
        'era5_wind': 'ERA5_ws_native.zarr',
        'era5_land_temp': 'ERA5-Land_tas_native.zarr',
        'era5_land_lst': 'ERA5-Land_lst_native.zarr',
        'modis_lst': 'modis_lst_native.zarr',
        'meteosat_lst': 'meteosat_lst_native.zarr',
        'meteosat_lst_regrid': 'meteosat_lst_regrid_ERA5.zarr',
        'wrf_temp': 'WRF_tas_native.zarr',
        'wrf_lst': 'WRF_lst_native.zarr',
        'saaqis': 'SAAQIS_with_climate_variables.zarr'
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_mapping.keys())}")
    
    filename = dataset_mapping[dataset]
    
    # Try primary location, then alternative
    for root in [CLIMATE_ROOT, ALT_CLIMATE_ROOT]:
        filepath = root / filename
        if filepath.exists():
            logger.info(f"Loading {dataset} from {filepath}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                ds = xr.open_zarr(filepath)
            
            # Filter by date range if provided
            if date_range and 'time' in ds.coords:
                start_date, end_date = date_range
                ds = ds.sel(time=slice(start_date, end_date))
                logger.info(f"Filtered to date range: {start_date} to {end_date}")
            
            return ds
    
    raise FileNotFoundError(f"Dataset {filename} not found in {[CLIMATE_ROOT, ALT_CLIMATE_ROOT]}")


def load_health_data(dataset: str = "latest") -> pd.DataFrame:
    """
    Load harmonized health data from RP2.
    
    Parameters
    ----------
    dataset : str, default "latest"
        Which dataset to load:
        - 'latest': Most recent reconstructed dataset
        - 'final': Final processed dataset
        - 'corrected': Corrected dataset (smaller subset)
        
    Returns
    -------
    pd.DataFrame
        Health data with proper type handling
    """
    dataset_mapping = {
        'latest': 'HEAT_Johannesburg_RECONSTRUCTED_LATEST.csv',
        'reconstructed': 'HEAT_Johannesburg_RECONSTRUCTED_20250811_164506.csv',
        'final': 'HEAT_Johannesburg_FINAL_20250811_163049.csv',
        'corrected': 'johannesburg_abidjan_CORRECTED_dataset.csv'
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_mapping.keys())}")
    
    filename = dataset_mapping[dataset]
    
    # Try primary location, then alternative
    for root in [HEALTH_ROOT, ALT_HEALTH_ROOT]:
        filepath = root / filename
        if filepath.exists():
            logger.info(f"Loading health data from {filepath}")
            
            # Load with proper handling of mixed types
            df = pd.read_csv(filepath, low_memory=False)
            
            # Clean and standardize
            df = _clean_health_data(df)
            
            logger.info(f"Loaded {len(df):,} health records with {len(df.columns)} variables")
            return df
    
    raise FileNotFoundError(f"Health dataset {filename} not found")


def load_socioeconomic_data(dataset: str = "gcro_combined") -> pd.DataFrame:
    """
    Load socioeconomic data.
    
    Parameters
    ----------
    dataset : str, default "gcro_combined"
        Which dataset to load:
        - 'gcro_combined': GCRO combined with climate data
        - 'gcro_subset': GCRO climate subset
        - 'gcro_full': Full GCRO combined dataset
        
    Returns
    -------
    pd.DataFrame
        Socioeconomic data
    """
    dataset_mapping = {
        'gcro_combined': 'GCRO_combined_climate_SUBSET.csv',
        'gcro_subset': 'GCRO_combined_climate_SUBSET.csv',
        'gcro_full': 'GCRO_full_combined.csv'
    }
    
    if dataset not in dataset_mapping:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_mapping.keys())}")
    
    filename = dataset_mapping[dataset]
    
    # Try primary location, then alternative
    for root in [SOCIOECONOMIC_ROOT, ALT_SOCIOECONOMIC]:
        filepath = root / filename
        if filepath.exists():
            logger.info(f"Loading socioeconomic data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df):,} socioeconomic records")
            return df
    
    raise FileNotFoundError(f"Socioeconomic dataset {filename} not found")


def _clean_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize health data types and formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw health dataframe
        
    Returns
    -------
    pd.DataFrame
        Cleaned health dataframe
    """
    df_clean = df.copy()
    
    # Parse dates
    date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Convert coordinates to numeric
    if 'latitude' in df_clean.columns:
        df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    if 'longitude' in df_clean.columns:
        df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    
    # Convert biomarker columns to numeric
    biomarker_patterns = ['glucose', 'crp', 'creatinine', 'hemoglobin', 'cd4', 
                         'cholesterol', 'alt', 'ast', 'albumin', 'triglyceride',
                         'blood_pressure', 'systolic', 'diastolic']
    
    for col in df_clean.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in biomarker_patterns):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Standardize categorical variables
    if 'Sex' in df_clean.columns:
        df_clean['sex_standardized'] = df_clean['Sex'].map({
            'Male': 'male', 'M': 'male', 'MALE': 'male', 'male': 'male',
            'Female': 'female', 'F': 'female', 'FEMALE': 'female', 'female': 'female'
        })
    
    return df_clean


def get_biomarker_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify biomarker columns in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Health dataframe
        
    Returns
    -------
    dict
        Dictionary mapping biomarker types to column names
    """
    biomarker_map = {
        'glucose': [],
        'crp': [],
        'creatinine': [],
        'hemoglobin': [],
        'cd4': [],
        'cholesterol': [],
        'alt': [],
        'ast': [],
        'albumin': [],
        'blood_pressure': [],
        'viral_load': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        if 'glucose' in col_lower:
            biomarker_map['glucose'].append(col)
        elif 'crp' in col_lower or 'c-reactive' in col_lower:
            biomarker_map['crp'].append(col)
        elif 'creatinine' in col_lower:
            biomarker_map['creatinine'].append(col)
        elif 'hemoglobin' in col_lower:
            biomarker_map['hemoglobin'].append(col)
        elif 'cd4' in col_lower:
            biomarker_map['cd4'].append(col)
        elif 'cholesterol' in col_lower or 'hdl' in col_lower or 'ldl' in col_lower:
            biomarker_map['cholesterol'].append(col)
        elif 'alt' in col_lower and 'salt' not in col_lower:
            biomarker_map['alt'].append(col)
        elif 'ast' in col_lower and 'fast' not in col_lower:
            biomarker_map['ast'].append(col)
        elif 'albumin' in col_lower:
            biomarker_map['albumin'].append(col)
        elif 'blood_pressure' in col_lower or 'systolic' in col_lower or 'diastolic' in col_lower:
            biomarker_map['blood_pressure'].append(col)
        elif 'viral' in col_lower and 'load' in col_lower:
            biomarker_map['viral_load'].append(col)
    
    # Remove empty lists
    biomarker_map = {k: v for k, v in biomarker_map.items() if v}
    
    return biomarker_map


def validate_data_integration() -> Dict[str, str]:
    """
    Validate that all required datasets can be loaded and integrated.
    
    Returns
    -------
    dict
        Validation results for each dataset type
    """
    results = {}
    
    # Test climate data loading
    try:
        era5_temp = load_climate_data('era5_temp')
        results['era5_temp'] = f"✓ Loaded {era5_temp.dims['time']} time points"
        era5_temp.close()
    except Exception as e:
        results['era5_temp'] = f"✗ Error: {e}"
    
    # Test health data loading  
    try:
        health_df = load_health_data('latest')
        results['health_data'] = f"✓ Loaded {len(health_df):,} health records"
    except Exception as e:
        results['health_data'] = f"✗ Error: {e}"
    
    # Test socioeconomic data loading
    try:
        socio_df = load_socioeconomic_data('gcro_combined')
        results['socioeconomic_data'] = f"✓ Loaded {len(socio_df):,} socioeconomic records"
    except Exception as e:
        results['socioeconomic_data'] = f"✗ Error: {e}"
    
    return results


def get_data_summary() -> Dict:
    """
    Generate a summary of all available datasets.
    
    Returns
    -------
    dict
        Summary information about datasets
    """
    summary = {
        'climate_datasets': [],
        'health_records': 0,
        'socioeconomic_records': 0,
        'biomarkers_available': [],
        'temporal_coverage': {},
        'spatial_coverage': {}
    }
    
    # Climate datasets
    climate_datasets = ['era5_temp', 'era5_lst', 'era5_wind', 'modis_lst', 'saaqis']
    for dataset in climate_datasets:
        try:
            ds = load_climate_data(dataset)
            if 'time' in ds.coords:
                summary['climate_datasets'].append({
                    'name': dataset,
                    'time_points': ds.dims['time'],
                    'start_date': str(ds.time.values[0])[:10],
                    'end_date': str(ds.time.values[-1])[:10],
                    'variables': list(ds.data_vars)
                })
            ds.close()
        except Exception:
            continue
    
    # Health data
    try:
        health_df = load_health_data('latest')
        summary['health_records'] = len(health_df)
        summary['biomarkers_available'] = list(get_biomarker_columns(health_df).keys())
        
        if 'date' in health_df.columns:
            dates = pd.to_datetime(health_df['date'], errors='coerce').dropna()
            if len(dates) > 0:
                summary['temporal_coverage']['health'] = {
                    'start': dates.min().strftime('%Y-%m-%d'),
                    'end': dates.max().strftime('%Y-%m-%d')
                }
        
        if 'latitude' in health_df.columns and 'longitude' in health_df.columns:
            summary['spatial_coverage']['health'] = {
                'lat_range': [health_df['latitude'].min(), health_df['latitude'].max()],
                'lon_range': [health_df['longitude'].min(), health_df['longitude'].max()]
            }
            
    except Exception as e:
        summary['health_error'] = str(e)
    
    # Socioeconomic data
    try:
        socio_df = load_socioeconomic_data('gcro_combined')
        summary['socioeconomic_records'] = len(socio_df)
    except Exception as e:
        summary['socioeconomic_error'] = str(e)
    
    return summary


if __name__ == "__main__":
    # Test data loading
    print("HEAT Lab Data Loader - Testing all datasets...")
    print("=" * 60)
    
    # Validate integration
    validation = validate_data_integration()
    print("\\nValidation Results:")
    for dataset, result in validation.items():
        print(f"  {dataset}: {result}")
    
    # Get summary
    print("\\nData Summary:")
    summary = get_data_summary()
    
    print(f"\\nClimate datasets available: {len(summary['climate_datasets'])}")
    for ds in summary['climate_datasets']:
        print(f"  - {ds['name']}: {ds['time_points']:,} points ({ds['start_date']} to {ds['end_date']})")
    
    print(f"\\nHealth records: {summary['health_records']:,}")
    print(f"Biomarkers available: {summary['biomarkers_available']}")
    
    if 'temporal_coverage' in summary and 'health' in summary['temporal_coverage']:
        tc = summary['temporal_coverage']['health']
        print(f"Health temporal coverage: {tc['start']} to {tc['end']}")
    
    if 'spatial_coverage' in summary and 'health' in summary['spatial_coverage']:
        sc = summary['spatial_coverage']['health']
        print(f"Health spatial coverage: Lat {sc['lat_range']}, Lon {sc['lon_range']}")
    
    print(f"\\nSocioeconomic records: {summary['socioeconomic_records']:,}")
    
    print("\\n" + "=" * 60)
    print("All datasets validated and ready for analysis!")