#!/usr/bin/env python3
"""
Test climate integration specifically with clinical data subset
"""

import pandas as pd
import numpy as np
from climate_data_integrator import ClimateDataIntegrator

def test_with_clinical_subset():
    """Test climate integration with clinical data that has coordinates."""

    print("🧪 Testing Climate Integration with Clinical Data Subset")
    print("=" * 60)

    # Load full dataset
    print("📊 Loading full dataset...")
    df = pd.read_csv("../data/MASTER_INTEGRATED_DATASET.csv", low_memory=False)
    print(f"   Total records: {len(df):,}")

    # Filter to clinical data with coordinates
    print("🏥 Filtering to clinical data with coordinates...")
    clinical_data = df[df['data_source'] == 'RP2_Clinical'].copy()
    print(f"   Clinical records: {len(clinical_data):,}")

    # Check coordinate availability
    lat_valid = pd.to_numeric(clinical_data['latitude'], errors='coerce').notna()
    lon_valid = pd.to_numeric(clinical_data['longitude'], errors='coerce').notna()
    date_valid = pd.to_datetime(clinical_data['primary_date'], errors='coerce').notna()

    valid_coords = lat_valid & lon_valid & date_valid
    valid_clinical = clinical_data[valid_coords].copy()

    print(f"   Records with valid coordinates & dates: {len(valid_clinical):,}")

    if len(valid_clinical) == 0:
        print("❌ No clinical records with valid coordinates found!")
        return None, None

    # Show sample coordinates and dates
    print(f"\n📍 Sample coordinate/date combinations:")
    sample_data = valid_clinical[['latitude', 'longitude', 'primary_date']].head()
    for idx, row in sample_data.iterrows():
        lat = pd.to_numeric(row['latitude'], errors='coerce')
        lon = pd.to_numeric(row['longitude'], errors='coerce')
        print(f"   Lat: {lat:.4f}, Lon: {lon:.4f}, Date: {row['primary_date']}")

    # Initialize climate integrator
    integrator = ClimateDataIntegrator()

    # Test with a small subset first (50 records)
    test_subset = valid_clinical.head(50).copy()
    print(f"\n🌡️  Testing climate integration with {len(test_subset)} records...")

    # Integrate climate data
    df_integrated = integrator.integrate_climate_with_health_data(test_subset)

    # Check results
    climate_cols = [col for col in df_integrated.columns if any(dataset in col for dataset in
                   ['ERA5', 'SAAQIS', 'meteosat', 'modis', 'WRF'])]

    print(f"✅ Climate integration results:")
    print(f"   Climate columns added: {len(climate_cols)}")
    print(f"   Integration report: {integrator.integration_report}")

    # Create summary
    summary = integrator.create_climate_summary(df_integrated)
    print(f"\n📋 Climate Data Summary:")
    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("   No climate data successfully integrated")

    # Save results
    df_integrated.to_csv("../data/clinical_climate_integration_test.csv", index=False)
    if len(summary) > 0:
        summary.to_csv("../data/clinical_climate_summary.csv", index=False)

    print(f"\n💾 Results saved:")
    print(f"   ../data/clinical_climate_integration_test.csv")
    if len(summary) > 0:
        print(f"   ../data/clinical_climate_summary.csv")

    return df_integrated, summary

def test_single_record_extraction():
    """Test extraction for a single record to debug the process."""

    print("\n🔬 Testing Single Record Climate Extraction")
    print("-" * 50)

    # Load clinical data
    df = pd.read_csv("../data/MASTER_INTEGRATED_DATASET.csv", low_memory=False)
    clinical_data = df[df['data_source'] == 'RP2_Clinical'].copy()

    # Get first valid record
    lat_valid = pd.to_numeric(clinical_data['latitude'], errors='coerce').notna()
    lon_valid = pd.to_numeric(clinical_data['longitude'], errors='coerce').notna()
    date_valid = pd.to_datetime(clinical_data['primary_date'], errors='coerce').notna()

    valid_records = clinical_data[lat_valid & lon_valid & date_valid]

    if len(valid_records) == 0:
        print("❌ No valid records for single extraction test")
        return

    test_record = valid_records.iloc[0]
    lat = pd.to_numeric(test_record['latitude'])
    lon = pd.to_numeric(test_record['longitude'])
    date = pd.to_datetime(test_record['primary_date'])

    print(f"📍 Test record:")
    print(f"   Latitude: {lat:.6f}")
    print(f"   Longitude: {lon:.6f}")
    print(f"   Date: {date}")

    # Initialize integrator and load one dataset
    integrator = ClimateDataIntegrator()
    climate_files = integrator.discover_climate_files()

    # Test with ERA5_tas_native (should be most reliable)
    dataset_name = 'ERA5_tas_native'
    if dataset_name in climate_files:
        print(f"\n🌡️  Testing extraction from {dataset_name}...")
        dataset = integrator.load_climate_dataset(climate_files[dataset_name], dataset_name)

        if dataset is not None:
            # Try extraction
            extracted_data = integrator.extract_point_data(dataset, lat, lon, date)
            print(f"   Extracted data: {extracted_data}")

            if not extracted_data:
                print("   ⚠️  No data extracted - investigating...")

                # Check spatial bounds
                if 'lat' in dataset.dims and 'lon' in dataset.dims:
                    lat_bounds = (dataset.lat.min().values, dataset.lat.max().values)
                    lon_bounds = (dataset.lon.min().values, dataset.lon.max().values)
                    print(f"   Dataset spatial bounds: Lat {lat_bounds}, Lon {lon_bounds}")
                    print(f"   Point within bounds: Lat {lat_bounds[0] <= lat <= lat_bounds[1]}, Lon {lon_bounds[0] <= lon <= lon_bounds[1]}")

                # Check temporal bounds
                if 'time' in dataset.dims:
                    time_bounds = (dataset.time.min().values, dataset.time.max().values)
                    print(f"   Dataset temporal bounds: {time_bounds}")
                    print(f"   Date within bounds: {time_bounds[0] <= date <= time_bounds[1]}")

if __name__ == "__main__":
    # Test with clinical subset
    df_integrated, summary = test_with_clinical_subset()

    # Test single record extraction for debugging
    test_single_record_extraction()