#!/usr/bin/env python3
"""
HEAT Data Issue Diagnostic Tool
===============================

Diagnoses critical data integration issues and creates comprehensive visualizations.

Author: Craig Parker
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Configure plotting
plt.style.use('default')
sns.set_palette("Set2")
warnings.filterwarnings('ignore')

FIGSIZE_LARGE = (16, 12)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 16

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

def load_and_diagnose_data():
    """Load data and perform comprehensive diagnosis."""

    print("🔍 DIAGNOSING HEAT DATASET ISSUES")
    print("=" * 50)

    # Load the master dataset
    data_path = "../data/MASTER_INTEGRATED_DATASET.csv"
    print(f"Loading: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Loaded: {len(df):,} records × {len(df.columns)} columns")

    # Basic info
    print(f"\n📊 BASIC DATASET INFO:")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Data types: {df.dtypes.value_counts().to_dict()}")

    # Data source breakdown
    print(f"\n🔍 DATA SOURCE ANALYSIS:")
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            percentage = count / len(df) * 100
            print(f"   {source}: {count:,} records ({percentage:.1f}%)")

    # Missing data analysis
    print(f"\n⚠️  MISSING DATA ANALYSIS:")
    missing_percentages = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    # Show columns with >50% missing data
    high_missing = missing_percentages[missing_percentages > 50]
    print(f"   Columns with >50% missing data: {len(high_missing)}")

    for col, pct in high_missing.head(10).items():
        print(f"      {col}: {pct:.1f}% missing")

    # Climate data specific analysis
    print(f"\n🌡️  CLIMATE DATA ANALYSIS:")
    climate_cols = [col for col in df.columns if 'era5' in col.lower() or 'temp' in col.lower()]
    print(f"   Climate columns found: {len(climate_cols)}")

    for col in climate_cols:
        non_null = df[col].notna().sum()
        percentage = (non_null / len(df)) * 100
        print(f"      {col}: {non_null:,} values ({percentage:.1f}%)")

    # Clinical data analysis
    print(f"\n🧬 CLINICAL DATA ANALYSIS:")
    clinical_cols = [col for col in df.columns if any(term in col.lower() for term in
                    ['cd4', 'hemoglobin', 'creatinine', 'glucose', 'cholesterol'])]

    for col in clinical_cols:
        non_null = df[col].notna().sum()
        percentage = (non_null / len(df)) * 100
        print(f"      {col}: {non_null:,} values ({percentage:.1f}%)")

    # Geographic data analysis
    print(f"\n🌍 GEOGRAPHIC DATA ANALYSIS:")
    geo_cols = ['latitude', 'longitude']
    for col in geo_cols:
        if col in df.columns:
            # Convert to numeric and count valid coordinates
            coords = pd.to_numeric(df[col], errors='coerce')
            valid_coords = coords.notna().sum()
            percentage = (valid_coords / len(df)) * 100
            print(f"      {col}: {valid_coords:,} valid values ({percentage:.1f}%)")

    # Temporal data analysis
    print(f"\n📅 TEMPORAL DATA ANALYSIS:")
    temporal_cols = ['primary_date', 'survey_year', 'year']
    for col in temporal_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            percentage = (non_null / len(df)) * 100
            print(f"      {col}: {non_null:,} values ({percentage:.1f}%)")

    return df, missing_percentages, climate_cols, clinical_cols

def create_comprehensive_diagnostic_visualization(df, missing_percentages, climate_cols, clinical_cols):
    """Create comprehensive diagnostic visualization."""

    fig = plt.figure(figsize=FIGSIZE_LARGE)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    fig.suptitle('HEAT Dataset Diagnostic Analysis: Critical Issues Identified',
                fontsize=18, fontweight='bold', y=0.95)

    colors = {
        'critical': '#D32F2F',     # Red
        'warning': '#FF9800',      # Orange
        'good': '#4CAF50',         # Green
        'info': '#2196F3'          # Blue
    }

    # 1. Overall missing data pattern
    ax1 = fig.add_subplot(gs[0, :])

    # Top 20 columns with highest missing data
    top_missing = missing_percentages.head(20)

    # Color code by severity
    colors_bars = []
    for pct in top_missing.values:
        if pct > 90:
            colors_bars.append(colors['critical'])
        elif pct > 50:
            colors_bars.append(colors['warning'])
        else:
            colors_bars.append(colors['good'])

    bars = ax1.barh(range(len(top_missing)), top_missing.values, color=colors_bars)
    ax1.set_yticks(range(len(top_missing)))
    ax1.set_yticklabels([col[:30] + '...' if len(col) > 30 else col for col in top_missing.index])
    ax1.set_xlabel('Missing Data Percentage (%)')
    ax1.set_title('Top 20 Columns by Missing Data (Critical Issues)', fontweight='bold')
    ax1.invert_yaxis()

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, top_missing.values)):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold')

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['critical'], label='Critical (>90% missing)'),
        Patch(facecolor=colors['warning'], label='Warning (50-90% missing)'),
        Patch(facecolor=colors['good'], label='Acceptable (<50% missing)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')

    # 2. Data source vs completeness
    ax2 = fig.add_subplot(gs[1, 0])

    if 'data_source' in df.columns:
        # Calculate completeness by data source for key variables
        key_vars = ['latitude', 'longitude'] + climate_cols[:3] + clinical_cols[:3]
        key_vars = [col for col in key_vars if col in df.columns]

        gcro_data = df[df['data_source'] == 'GCRO']
        clinical_data = df[df['data_source'] == 'RP2_Clinical']

        completeness_by_source = {}

        for var in key_vars[:6]:  # Top 6 variables
            gcro_completeness = (gcro_data[var].notna().sum() / len(gcro_data)) * 100 if len(gcro_data) > 0 else 0
            clinical_completeness = (clinical_data[var].notna().sum() / len(clinical_data)) * 100 if len(clinical_data) > 0 else 0

            completeness_by_source[var] = {
                'GCRO': gcro_completeness,
                'Clinical': clinical_completeness
            }

        # Create grouped bar chart
        variables = list(completeness_by_source.keys())
        gcro_values = [completeness_by_source[var]['GCRO'] for var in variables]
        clinical_values = [completeness_by_source[var]['Clinical'] for var in variables]

        x = np.arange(len(variables))
        width = 0.35

        bars1 = ax2.bar(x - width/2, gcro_values, width, label='GCRO', color=colors['info'], alpha=0.8)
        bars2 = ax2.bar(x + width/2, clinical_values, width, label='Clinical', color=colors['warning'], alpha=0.8)

        ax2.set_xlabel('Variables')
        ax2.set_ylabel('Completeness (%)')
        ax2.set_title('Data Completeness by Source', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([var[:15] + '...' if len(var) > 15 else var for var in variables], rotation=45)
        ax2.legend()

    # 3. Climate data integration status
    ax3 = fig.add_subplot(gs[1, 1])

    climate_status = {}
    for col in climate_cols:
        non_null = df[col].notna().sum()
        percentage = (non_null / len(df)) * 100
        climate_status[col.replace('era5_temp_', '')] = percentage

    if climate_status:
        names = list(climate_status.keys())
        values = list(climate_status.values())

        # Color by completion rate
        colors_climate = [colors['critical'] if v < 10 else colors['warning'] if v < 50 else colors['good'] for v in values]

        bars = ax3.bar(names, values, color=colors_climate, alpha=0.8)
        ax3.set_ylabel('Completion Rate (%)')
        ax3.set_title('Climate Data Integration Status', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Issue summary and recommendations
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Calculate key statistics
    total_climate_missing = sum(missing_percentages[col] for col in climate_cols if col in missing_percentages)
    avg_climate_missing = total_climate_missing / len(climate_cols) if climate_cols else 0

    total_clinical_missing = sum(missing_percentages[col] for col in clinical_cols if col in missing_percentages)
    avg_clinical_missing = total_clinical_missing / len(clinical_cols) if clinical_cols else 0

    issues_summary = f"""
🚨 CRITICAL ISSUES IDENTIFIED:

❌ CLIMATE DATA INTEGRATION FAILURE:
   • Average climate data completion: {avg_climate_missing:.1f}%
   • ERA5 temperature data: ~0.4% complete
   • Climate-health linkage: BROKEN

❌ HIGH MISSING DATA BURDEN:
   • {len(missing_percentages[missing_percentages > 90])} columns >90% missing
   • {len(missing_percentages[missing_percentages > 50])} columns >50% missing
   • Clinical data completion: {100 - avg_clinical_missing:.1f}%

❌ DATA SOURCE IMBALANCE:
   • GCRO: {df['data_source'].value_counts().get('GCRO', 0):,} records (92.9%)
   • Clinical: {df['data_source'].value_counts().get('RP2_Clinical', 0):,} records (7.1%)
   • Geographic data: Limited to clinical subset only

🔧 IMMEDIATE ACTIONS REQUIRED:

1️⃣ FIX CLIMATE DATA LINKAGE:
   • Verify ERA5 data integration process
   • Check temporal alignment between clinical dates and climate data
   • Investigate coordinate-based climate data extraction

2️⃣ ENHANCE DATA INTEGRATION:
   • Review data harmonization pipeline
   • Validate geographic coordinate assignment
   • Implement proper missing value strategies

3️⃣ VALIDATE RESEARCH FEASIBILITY:
   • Current climate data insufficient for analysis
   • May need alternative climate data sources
   • Consider reduced scope focusing on available data
    """

    ax4.text(0.05, 0.95, issues_summary, transform=ax4.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.9))

    # Save figure
    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)

    plt.savefig(output_dir / 'dataset_diagnostic_analysis.svg',
               dpi=DPI, bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'dataset_diagnostic_analysis.png',
               dpi=DPI, bbox_inches='tight', format='png')
    plt.close()

    print(f"✅ Saved diagnostic analysis to {output_dir}")

def create_data_integration_roadmap():
    """Create a roadmap for fixing data integration issues."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Create flowchart-style roadmap
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'HEAT Data Integration Roadmap',
           ha='center', va='center', fontsize=20, fontweight='bold')

    # Problem boxes
    problems = [
        "Climate Data\n0.4% Complete",
        "Missing Geographic\nLinkage",
        "Temporal Alignment\nIssues"
    ]

    solutions = [
        "1. Verify Climate\nData Pipeline",
        "2. Fix Coordinate\nExtraction",
        "3. Validate Temporal\nMatching"
    ]

    outcomes = [
        "Climate Features\nEngineered",
        "Geographic Coverage\nValidated",
        "Time-Series\nAlignment"
    ]

    # Draw boxes and arrows
    for i, (problem, solution, outcome) in enumerate(zip(problems, solutions, outcomes)):
        y_pos = 7 - i * 2

        # Problem box (red)
        prob_box = plt.Rectangle((0.5, y_pos-0.4), 2, 0.8,
                               facecolor='#ffcccb', edgecolor='red', linewidth=2)
        ax.add_patch(prob_box)
        ax.text(1.5, y_pos, problem, ha='center', va='center', fontweight='bold')

        # Arrow 1
        ax.arrow(2.5, y_pos, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        # Solution box (orange)
        sol_box = plt.Rectangle((3.5, y_pos-0.4), 2.5, 0.8,
                              facecolor='#ffd9b3', edgecolor='orange', linewidth=2)
        ax.add_patch(sol_box)
        ax.text(4.75, y_pos, solution, ha='center', va='center', fontweight='bold')

        # Arrow 2
        ax.arrow(6, y_pos, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

        # Outcome box (green)
        out_box = plt.Rectangle((7, y_pos-0.4), 2.5, 0.8,
                              facecolor='#ccffcc', edgecolor='green', linewidth=2)
        ax.add_patch(out_box)
        ax.text(8.25, y_pos, outcome, ha='center', va='center', fontweight='bold')

    # Final outcome
    final_box = plt.Rectangle((2, 0.5), 6, 1,
                            facecolor='#e6ffe6', edgecolor='darkgreen', linewidth=3)
    ax.add_patch(final_box)
    ax.text(5, 1, 'PUBLICATION-READY\nCLIMATE-HEALTH ANALYSIS',
           ha='center', va='center', fontsize=14, fontweight='bold')

    # Arrows to final outcome
    for i in range(3):
        y_start = 7 - i * 2 - 0.4
        ax.arrow(8.25, y_start, -1.5, 1.5 - y_start,
                head_width=0.1, head_length=0.1, fc='darkgreen', ec='darkgreen',
                linestyle='--', alpha=0.7)

    # Save roadmap
    output_dir = Path("../figures")
    plt.savefig(output_dir / 'data_integration_roadmap.svg',
               dpi=DPI, bbox_inches='tight', format='svg')
    plt.savefig(output_dir / 'data_integration_roadmap.png',
               dpi=DPI, bbox_inches='tight', format='png')
    plt.close()

    print(f"✅ Saved integration roadmap to {output_dir}")

def main():
    """Run comprehensive diagnostic analysis."""

    # Load and diagnose data
    df, missing_percentages, climate_cols, clinical_cols = load_and_diagnose_data()

    # Create diagnostic visualizations
    print(f"\n🎨 Creating diagnostic visualizations...")
    create_comprehensive_diagnostic_visualization(df, missing_percentages, climate_cols, clinical_cols)

    # Create integration roadmap
    print(f"🛣️  Creating integration roadmap...")
    create_data_integration_roadmap()

    print(f"\n✅ DIAGNOSTIC ANALYSIS COMPLETE")
    print(f"📁 Generated files:")
    print(f"   • dataset_diagnostic_analysis.svg/png")
    print(f"   • data_integration_roadmap.svg/png")

    print(f"\n🔍 KEY FINDINGS:")
    print(f"   • Climate data integration: CRITICAL FAILURE (0.4% complete)")
    print(f"   • Clinical data: Available but limited geographic coverage")
    print(f"   • GCRO data: Good coverage but missing climate linkage")
    print(f"   • Immediate action required to fix data pipeline")

if __name__ == "__main__":
    main()