#!/usr/bin/env python3
"""
Analysis of the MASTER_INTEGRATED_DATASET
==========================================

This script analyzes the existing integrated dataset to understand:
1. What climate data is already integrated
2. Data coverage and quality
3. Geographic and temporal distribution
4. Suitability for heat-health ML analysis

Author: HEAT Research Team
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

def analyze_master_dataset():
    """Comprehensive analysis of the master integrated dataset."""

    print("🔍 ANALYZING MASTER INTEGRATED DATASET")
    print("="*60)

    # Load the dataset in chunks to find coordinate data
    data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/MASTER_INTEGRATED_DATASET.csv"

    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"  Total records: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")

    # Basic data composition
    print("\n📈 Data Composition:")
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        print("  Data sources:")
        for source, count in source_counts.items():
            print(f"    {source}: {count:,} records ({count/len(df)*100:.1f}%)")

    # Find coordinate data
    print("\n🌍 Geographic Coverage:")
    lat_col = 'latitude'
    lon_col = 'longitude'

    coords_available = df[lat_col].notna() & df[lon_col].notna()
    coords_count = coords_available.sum()

    print(f"  Records with coordinates: {coords_count:,}/{len(df):,} ({coords_count/len(df)*100:.1f}%)")

    if coords_count > 0:
        # Analyze coordinate data
        coords_df = df[coords_available].copy()

        # Convert coordinates to numeric
        coords_df[lat_col] = pd.to_numeric(coords_df[lat_col], errors='coerce')
        coords_df[lon_col] = pd.to_numeric(coords_df[lon_col], errors='coerce')

        # Remove any rows with invalid coordinates
        coords_df = coords_df[coords_df[lat_col].notna() & coords_df[lon_col].notna()]

        if len(coords_df) > 0:
            print(f"  Latitude range: [{coords_df[lat_col].min():.6f}, {coords_df[lat_col].max():.6f}]")
            print(f"  Longitude range: [{coords_df[lon_col].min():.6f}, {coords_df[lon_col].max():.6f}]")

        # Show sample coordinates
        print("\n  Sample coordinates:")
        sample_coords = coords_df[[lat_col, lon_col, 'data_source']].head(5)
        for _, row in sample_coords.iterrows():
            print(f"    Lat: {row[lat_col]:.6f}, Lon: {row[lon_col]:.6f} ({row['data_source']})")

    # Analyze climate variables
    print("\n🌡️  Climate Variables Analysis:")

    # Find climate columns
    climate_cols = []
    climate_patterns = ['era5', 'temp', 'climate', 'weather', 'heat', 'lst', 'tas']

    for col in df.columns:
        if any(pattern in col.lower() for pattern in climate_patterns):
            climate_cols.append(col)

    print(f"  Climate columns found: {len(climate_cols)}")

    # Analyze each climate variable
    climate_summary = []

    for col in climate_cols:
        values = pd.to_numeric(df[col], errors='coerce')
        non_null = values.notna().sum()

        if non_null > 0:
            climate_summary.append({
                'Variable': col,
                'Count': non_null,
                'Completion_%': (non_null / len(df)) * 100,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Q25': values.quantile(0.25),
                'Q75': values.quantile(0.75)
            })

    if climate_summary:
        climate_df = pd.DataFrame(climate_summary)
        climate_df = climate_df.sort_values('Completion_%', ascending=False)

        print(f"\n  Top 10 climate variables by completion:")
        top_climate = climate_df.head(10)[['Variable', 'Count', 'Completion_%', 'Mean', 'Min', 'Max']]
        top_climate = top_climate.round(2)

        for _, row in top_climate.iterrows():
            print(f"    {row['Variable'][:40]:40s} | {row['Count']:8,} | {row['Completion_%']:6.1f}% | "
                  f"Mean: {row['Mean']:8.2f} | Range: [{row['Min']:6.2f}, {row['Max']:6.2f}]")

    # Analyze temporal coverage
    print("\n📅 Temporal Coverage:")
    date_cols = [col for col in df.columns if 'date' in col.lower() or col in ['year', 'month', 'survey_year']]

    for date_col in date_cols[:3]:  # Check first 3 date columns
        if date_col in df.columns:
            non_null_dates = df[date_col].notna().sum()
            print(f"  {date_col}: {non_null_dates:,} non-null values")

            if non_null_dates > 0:
                if 'year' in date_col.lower():
                    year_values = pd.to_numeric(df[date_col], errors='coerce').dropna()
                    if len(year_values) > 0:
                        print(f"    Year range: {int(year_values.min())} - {int(year_values.max())}")
                else:
                    try:
                        date_values = pd.to_datetime(df[date_col], errors='coerce').dropna()
                        if len(date_values) > 0:
                            print(f"    Date range: {date_values.min().date()} to {date_values.max().date()}")
                    except:
                        print(f"    Could not parse dates")

    # Analyze biomarkers for heat-health analysis
    print("\n🧬 Health Biomarkers for Heat-Health Analysis:")

    biomarker_patterns = ['cd4', 'viral', 'hemoglobin', 'hematocrit', 'creatinine', 'blood pressure', 'heart rate', 'temperature']
    biomarker_cols = []

    for col in df.columns:
        if any(pattern in col.lower() for pattern in biomarker_patterns):
            biomarker_cols.append(col)

    print(f"  Biomarker columns found: {len(biomarker_cols)}")

    biomarker_summary = []
    for col in biomarker_cols[:10]:  # Top 10 biomarkers
        values = pd.to_numeric(df[col], errors='coerce')
        non_null = values.notna().sum()

        if non_null > 0:
            biomarker_summary.append({
                'Biomarker': col,
                'Count': non_null,
                'Completion_%': (non_null / len(df)) * 100,
                'Mean': values.mean(),
                'Std': values.std()
            })

    if biomarker_summary:
        biomarker_df = pd.DataFrame(biomarker_summary)
        biomarker_df = biomarker_df.sort_values('Completion_%', ascending=False)

        print(f"\n  Top biomarkers by completion:")
        for _, row in biomarker_df.head(5).iterrows():
            print(f"    {row['Biomarker'][:40]:40s} | {row['Count']:8,} | {row['Completion_%']:6.1f}% | "
                  f"Mean: {row['Mean']:8.2f}")

    # Data readiness assessment
    print("\n✅ HEAT-HEALTH ML READINESS ASSESSMENT:")
    print("="*60)

    # Check for essential components
    has_coordinates = coords_count > 0
    has_climate_data = len(climate_cols) > 0 and any('era5' in col.lower() for col in climate_cols)
    has_health_outcomes = len(biomarker_cols) > 0
    has_temporal_data = any(df[col].notna().sum() > 100 for col in date_cols) if date_cols else False

    print(f"  ✓ Geographic coordinates: {'YES' if has_coordinates else 'NO'} ({coords_count:,} records)")
    print(f"  ✓ Climate variables: {'YES' if has_climate_data else 'NO'} ({len(climate_cols)} variables)")
    print(f"  ✓ Health biomarkers: {'YES' if has_health_outcomes else 'NO'} ({len(biomarker_cols)} variables)")
    print(f"  ✓ Temporal coverage: {'YES' if has_temporal_data else 'NO'}")

    overall_readiness = has_coordinates and has_climate_data and has_health_outcomes and has_temporal_data
    print(f"\n  🎯 Overall ML Readiness: {'READY' if overall_readiness else 'NEEDS WORK'}")

    if not overall_readiness:
        print("\n  🔧 Recommendations:")
        if not has_coordinates:
            print("    • Need to add geographic coordinates for spatial climate linking")
        if not has_climate_data:
            print("    • Need to integrate climate variables from zarr datasets")
        if not has_health_outcomes:
            print("    • Need to add health outcome variables")
        if not has_temporal_data:
            print("    • Need to improve temporal data coverage")

    # Save analysis results
    print("\n💾 Saving Analysis Results...")

    # Create summary report (convert numpy types to Python types)
    analysis_report = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'dataset_summary': {
            'total_records': int(len(df)),
            'total_columns': int(len(df.columns)),
            'records_with_coordinates': int(coords_count),
            'coordinate_completion_rate': float(coords_count / len(df) * 100)
        },
        'climate_variables': {
            'total_climate_columns': int(len(climate_cols)),
            'variables_with_data': int(len(climate_summary))
        },
        'health_variables': {
            'total_biomarker_columns': int(len(biomarker_cols)),
            'variables_with_data': int(len(biomarker_summary))
        },
        'ml_readiness': {
            'has_coordinates': bool(has_coordinates),
            'has_climate_data': bool(has_climate_data),
            'has_health_outcomes': bool(has_health_outcomes),
            'has_temporal_data': bool(has_temporal_data),
            'overall_ready': bool(overall_readiness)
        }
    }

    # Save results
    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)

    # Save analysis report
    with open(output_dir / "master_dataset_analysis.json", 'w') as f:
        json.dump(analysis_report, f, indent=2)

    # Save climate summary
    if climate_summary:
        climate_df.to_csv(output_dir / "master_dataset_climate_summary.csv", index=False)
        print(f"  ✓ Climate summary saved: {len(climate_summary)} variables")

    # Save biomarker summary
    if biomarker_summary:
        biomarker_df.to_csv(output_dir / "master_dataset_biomarker_summary.csv", index=False)
        print(f"  ✓ Biomarker summary saved: {len(biomarker_summary)} variables")

    print(f"\n✅ Analysis complete! Results saved to ../data/")

    return analysis_report, climate_df if climate_summary else None, biomarker_df if biomarker_summary else None

def create_data_overview_visualization():
    """Create visualizations of the data overview."""

    print("\n📊 Creating Data Overview Visualizations...")

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # Load analysis results
    output_dir = Path("../data")

    # Load climate summary if available
    climate_file = output_dir / "master_dataset_climate_summary.csv"
    if climate_file.exists():
        climate_df = pd.read_csv(climate_file)

        # Create climate variables completion plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Top 15 climate variables by completion rate
        top_climate = climate_df.head(15)

        # Completion rate plot
        bars1 = ax1.barh(range(len(top_climate)), top_climate['Completion_%'])
        ax1.set_yticks(range(len(top_climate)))
        ax1.set_yticklabels([var[:25] + '...' if len(var) > 25 else var for var in top_climate['Variable']], fontsize=10)
        ax1.set_xlabel('Completion Rate (%)')
        ax1.set_title('Climate Variables: Data Completion Rates', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        # Add completion rate labels
        for i, (_, row) in enumerate(top_climate.iterrows()):
            ax1.text(row['Completion_%'] + 1, i, f"{row['Completion_%']:.1f}%",
                    va='center', fontsize=9)

        # Value distribution plot for top variables
        top_5_climate = climate_df.head(5)
        ax2.scatter(top_5_climate['Mean'], top_5_climate['Std'],
                   s=top_5_climate['Completion_%']*3, alpha=0.6, c=range(len(top_5_climate)))

        for i, (_, row) in enumerate(top_5_climate.iterrows()):
            ax2.annotate(row['Variable'][:20],
                        (row['Mean'], row['Std']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax2.set_xlabel('Mean Value')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Climate Variables: Mean vs Variability\n(Bubble size = Completion %)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir.parent / "figures" / "climate_variables_overview.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir.parent / "figures" / "climate_variables_overview.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Climate variables overview saved")

    # Load biomarker summary if available
    biomarker_file = output_dir / "master_dataset_biomarker_summary.csv"
    if biomarker_file.exists():
        biomarker_df = pd.read_csv(biomarker_file)

        # Create biomarker overview
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Top 12 biomarkers
        top_biomarkers = biomarker_df.head(12)

        bars = ax.barh(range(len(top_biomarkers)), top_biomarkers['Completion_%'])
        ax.set_yticks(range(len(top_biomarkers)))
        ax.set_yticklabels([var[:30] + '...' if len(var) > 30 else var for var in top_biomarkers['Biomarker']], fontsize=11)
        ax.set_xlabel('Completion Rate (%)')
        ax.set_title('Health Biomarkers: Data Completion Rates', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add completion rate labels
        for i, (_, row) in enumerate(top_biomarkers.iterrows()):
            ax.text(row['Completion_%'] + 1, i, f"{row['Completion_%']:.1f}%",
                   va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir.parent / "figures" / "biomarkers_overview.svg", format='svg', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir.parent / "figures" / "biomarkers_overview.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✓ Biomarkers overview saved")

    print("  ✓ All visualizations saved to ../figures/")

def main():
    """Main execution function."""

    # Create figures directory
    figures_dir = Path("../figures")
    figures_dir.mkdir(exist_ok=True)

    # Analyze the master dataset
    analysis_report, climate_df, biomarker_df = analyze_master_dataset()

    # Create visualizations
    create_data_overview_visualization()

    return analysis_report, climate_df, biomarker_df

if __name__ == "__main__":
    analysis_report, climate_df, biomarker_df = main()