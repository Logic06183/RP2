#!/usr/bin/env python3
"""
HEAT Pipeline Visualization Creator
==================================

Creates publication-quality visualizations for data validation and analysis.

Author: Craig Parker
Date: 2025-09-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Import pipeline modules
import importlib.util
spec = importlib.util.spec_from_file_location("data_loader", "01_data_loader.py")
data_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader_module)
HeatDataLoader = data_loader_module.HeatDataLoader

# Configure plotting
plt.style.use('default')
sns.set_palette("Set2")
warnings.filterwarnings('ignore')

# Publication-quality styling
FIGSIZE_LARGE = (16, 12)
FIGSIZE_MEDIUM = (12, 8)
FIGSIZE_SMALL = (8, 6)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 16
LABEL_SIZE = 14

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
    'figure.titlesize': TITLE_SIZE + 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

class HeatVisualizationCreator:
    """Creates comprehensive visualizations for HEAT data analysis."""

    def __init__(self, output_dir: str = "../figures"):
        """Initialize the visualization creator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Color palette for scientific publication
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',     # Purple
            'accent': '#F18F01',        # Orange
            'success': '#C73E1D',       # Red
            'warning': '#F7DC6F',       # Yellow
            'neutral': '#7D8491',       # Gray
            'gcro': '#2E86AB',
            'clinical': '#A23B72',
            'climate': '#F18F01'
        }

    def create_dataset_overview(self, validation_report: dict, df: pd.DataFrame):
        """Create comprehensive dataset overview visualization."""
        fig = plt.figure(figsize=FIGSIZE_LARGE)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle('HEAT Dataset Overview: Integrated Climate-Health Analysis',
                    fontsize=TITLE_SIZE + 4, fontweight='bold', y=0.95)

        # 1. Data source composition (pie chart)
        ax1 = fig.add_subplot(gs[0, 0])
        data_sources = validation_report['data_sources']
        colors_pie = [self.colors['gcro'], self.colors['clinical']]

        wedges, texts, autotexts = ax1.pie(
            data_sources.values(),
            labels=data_sources.keys(),
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': FONT_SIZE}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax1.set_title('Data Source Composition\n(n = 128,465)', fontweight='bold')

        # 2. Temporal coverage
        ax2 = fig.add_subplot(gs[0, 1:])

        # GCRO survey years
        gcro_years = validation_report.get('gcro_survey_years', [])
        gcro_counts = [data_sources['GCRO'] / len(gcro_years)] * len(gcro_years)  # Approximate

        # Clinical years (sample by extracting from date range)
        clinical_range = validation_report.get('clinical_date_range', {})
        clinical_years = list(range(2002, 2022))  # Based on date range
        clinical_counts = [data_sources['RP2_Clinical'] / len(clinical_years)] * len(clinical_years)

        # Plot timeline
        ax2.bar([y - 0.2 for y in gcro_years], gcro_counts, width=0.4,
               color=self.colors['gcro'], label='GCRO Surveys', alpha=0.8)
        ax2.bar([y + 0.2 for y in clinical_years], clinical_counts, width=0.4,
               color=self.colors['clinical'], label='Clinical Data', alpha=0.8)

        ax2.set_xlabel('Year')
        ax2.set_ylabel('Approximate Records')
        ax2.set_title('Temporal Coverage (2002-2021)', fontweight='bold')
        ax2.legend()
        ax2.set_xlim(2001, 2022)

        # 3. Geographic coverage
        ax3 = fig.add_subplot(gs[1, 0])

        # Extract coordinates
        if 'geographic_coverage' in validation_report and 'error' not in validation_report['geographic_coverage']:
            geo = validation_report['geographic_coverage']

            # Create a simple map representation
            lat_range = geo['latitude_range']
            lon_range = geo['longitude_range']

            # Plot Johannesburg boundary approximation
            rect = patches.Rectangle(
                (lon_range[0], lat_range[0]),
                lon_range[1] - lon_range[0],
                lat_range[1] - lat_range[0],
                linewidth=2, edgecolor=self.colors['primary'],
                facecolor=self.colors['primary'], alpha=0.3
            )
            ax3.add_patch(rect)

            ax3.set_xlim(lon_range[0] - 0.01, lon_range[1] + 0.01)
            ax3.set_ylim(lat_range[0] - 0.01, lat_range[1] + 0.01)
            ax3.set_xlabel('Longitude')
            ax3.set_ylabel('Latitude')
            ax3.set_title('Geographic Coverage\nJohannesburg Metropolitan Area', fontweight='bold')

            # Add coordinate info
            ax3.text(0.05, 0.95, f"Lat: {lat_range[0]:.3f} to {lat_range[1]:.3f}°S\nLon: {lon_range[0]:.3f} to {lon_range[1]:.3f}°E",
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Biomarker availability
        ax4 = fig.add_subplot(gs[1, 1:])

        if 'clinical_biomarkers' in validation_report and 'error' not in validation_report['clinical_biomarkers']:
            biomarkers = validation_report['clinical_biomarkers']

            biomarker_names = list(biomarkers.keys())
            biomarker_counts = [biomarkers[bio]['non_null_count'] for bio in biomarker_names]
            completion_rates = [biomarkers[bio]['completion_rate'] * 100 for bio in biomarker_names]

            # Shorten biomarker names for display
            display_names = [name.split('(')[0].strip() for name in biomarker_names]

            # Create horizontal bar chart
            y_pos = np.arange(len(display_names))
            bars = ax4.barh(y_pos, biomarker_counts, color=self.colors['clinical'], alpha=0.8)

            # Add completion percentages
            for i, (bar, rate) in enumerate(zip(bars, completion_rates)):
                width = bar.get_width()
                ax4.text(width + 50, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1f}%', ha='left', va='center', fontweight='bold')

            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(display_names)
            ax4.set_xlabel('Number of Records')
            ax4.set_title('Clinical Biomarker Availability', fontweight='bold')
            ax4.invert_yaxis()

        # 5. Data quality summary
        ax5 = fig.add_subplot(gs[2, :])

        # Create summary statistics table
        summary_data = [
            ['Total Records', f"{validation_report['total_records']:,}"],
            ['Total Columns', f"{validation_report['total_columns']}"],
            ['Memory Usage', f"{validation_report['memory_usage_mb']:.1f} MB"],
            ['GCRO Records', f"{data_sources.get('GCRO', 0):,} ({data_sources.get('GCRO', 0) / validation_report['total_records'] * 100:.1f}%)"],
            ['Clinical Records', f"{data_sources.get('RP2_Clinical', 0):,} ({data_sources.get('RP2_Clinical', 0) / validation_report['total_records'] * 100:.1f}%)"],
            ['Valid Coordinates', f"{validation_report.get('geographic_coverage', {}).get('valid_coordinates', 'N/A'):,}"],
            ['Temporal Span', '19 years (2002-2021)']
        ]

        ax5.axis('tight')
        ax5.axis('off')

        table = ax5.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.3, 0.7])

        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZE)
        table.scale(1, 2)

        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(self.colors['primary'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

        ax5.set_title('Dataset Summary Statistics', fontweight='bold', pad=20)

        # Save figure
        plt.savefig(self.output_dir / 'dataset_overview.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'dataset_overview.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved dataset overview to {self.output_dir}")

    def create_climate_data_analysis(self, df: pd.DataFrame):
        """Analyze and visualize climate data coverage issues."""
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
        fig.suptitle('Climate Data Integration Analysis', fontsize=TITLE_SIZE + 2, fontweight='bold')

        # Identify climate columns
        climate_cols = [col for col in df.columns if 'era5' in col.lower()]

        # 1. Climate data availability by source
        ax1 = axes[0, 0]

        climate_availability = {}
        for col in climate_cols:
            total_records = len(df)
            non_null = df[col].notna().sum()
            climate_availability[col.replace('era5_temp_', '')] = (non_null / total_records) * 100

        if climate_availability:
            names = list(climate_availability.keys())
            values = list(climate_availability.values())

            bars = ax1.bar(names, values, color=self.colors['climate'], alpha=0.8)
            ax1.set_ylabel('Completion Rate (%)')
            ax1.set_title('Climate Data Completeness', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Data source vs climate availability
        ax2 = axes[0, 1]

        # Check climate data by data source
        gcro_data = df[df['data_source'] == 'GCRO']
        clinical_data = df[df['data_source'] == 'RP2_Clinical']

        source_climate_availability = {}
        for source_name, source_df in [('GCRO', gcro_data), ('Clinical', clinical_data)]:
            if len(source_df) > 0 and climate_cols:
                # Check first climate column as representative
                first_climate_col = climate_cols[0]
                if first_climate_col in source_df.columns:
                    availability = (source_df[first_climate_col].notna().sum() / len(source_df)) * 100
                    source_climate_availability[source_name] = availability

        if source_climate_availability:
            sources = list(source_climate_availability.keys())
            availabilities = list(source_climate_availability.values())
            colors = [self.colors['gcro'] if s == 'GCRO' else self.colors['clinical'] for s in sources]

            bars = ax2.bar(sources, availabilities, color=colors, alpha=0.8)
            ax2.set_ylabel('Climate Data Availability (%)')
            ax2.set_title('Climate Data by Source', fontweight='bold')

            # Add value labels
            for bar, value in zip(bars, availabilities):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Temporal pattern of climate data
        ax3 = axes[1, 0]

        # Check if we have date information with climate data
        if 'primary_date' in df.columns and climate_cols:
            df_with_dates = df.dropna(subset=['primary_date'])
            if len(df_with_dates) > 0:
                df_with_dates['date_parsed'] = pd.to_datetime(df_with_dates['primary_date'], errors='coerce')
                df_with_dates = df_with_dates.dropna(subset=['date_parsed'])

                if len(df_with_dates) > 0:
                    df_with_dates['year'] = df_with_dates['date_parsed'].dt.year

                    # Check climate availability by year
                    yearly_climate = df_with_dates.groupby('year').agg({
                        climate_cols[0]: lambda x: x.notna().sum() / len(x) * 100
                    }).reset_index()

                    ax3.plot(yearly_climate['year'], yearly_climate[climate_cols[0]],
                           marker='o', linewidth=2, markersize=6, color=self.colors['climate'])
                    ax3.set_xlabel('Year')
                    ax3.set_ylabel('Climate Data Availability (%)')
                    ax3.set_title('Climate Data Temporal Coverage', fontweight='bold')

        # 4. Issue identification summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Create issue summary text
        issues = []
        if climate_availability:
            max_completion = max(climate_availability.values())
            if max_completion < 50:
                issues.append(f"⚠️ Low climate data completion: {max_completion:.1f}% max")

            if len(set(climate_availability.values())) > 1:
                issues.append("⚠️ Inconsistent completion across climate variables")

        if len(climate_cols) < 10:
            issues.append(f"⚠️ Limited climate features: {len(climate_cols)} variables")

        # Add recommendations
        recommendations = [
            "✅ Verify climate data linkage process",
            "✅ Check temporal alignment between datasets",
            "✅ Investigate missing value patterns",
            "✅ Consider alternative climate data sources"
        ]

        summary_text = "CLIMATE DATA ISSUES IDENTIFIED:\n\n"
        summary_text += "\n".join(issues) if issues else "✅ No major issues detected"
        summary_text += "\n\nRECOMMENDATIONS:\n\n"
        summary_text += "\n".join(recommendations)

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=FONT_SIZE,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

        ax4.set_title('Climate Data Assessment', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'climate_data_analysis.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'climate_data_analysis.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved climate data analysis to {self.output_dir}")

    def create_gcro_variable_analysis(self, df: pd.DataFrame):
        """Analyze GCRO variables and provide selection rationale."""

        # Focus on GCRO data only
        gcro_data = df[df['data_source'] == 'GCRO'].copy()

        if len(gcro_data) == 0:
            print("⚠️ No GCRO data found for analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
        fig.suptitle('GCRO Variable Selection Analysis', fontsize=TITLE_SIZE + 2, fontweight='bold')

        # Key GCRO variables to analyze
        key_variables = {
            'age': 'Age (continuous)',
            'sex': 'Sex/Gender (categorical)',
            'race': 'Race/Ethnicity (categorical)',
            'education': 'Education Level (ordinal)',
            'employment': 'Employment Status (categorical)',
            'income': 'Income Level (ordinal)',
            'ward': 'Geographic Ward (categorical)',
            'municipality': 'Municipality (categorical)'
        }

        # 1. Variable completeness analysis
        ax1 = axes[0, 0]

        variable_completeness = {}
        for var, description in key_variables.items():
            if var in gcro_data.columns:
                completion_rate = (gcro_data[var].notna().sum() / len(gcro_data)) * 100
                variable_completeness[var] = completion_rate

        if variable_completeness:
            vars_sorted = sorted(variable_completeness.items(), key=lambda x: x[1], reverse=True)
            var_names = [v[0] for v in vars_sorted]
            completion_rates = [v[1] for v in vars_sorted]

            bars = ax1.barh(var_names, completion_rates, color=self.colors['gcro'], alpha=0.8)
            ax1.set_xlabel('Completion Rate (%)')
            ax1.set_title('GCRO Variable Completeness', fontweight='bold')

            # Add completion rate labels
            for bar, rate in zip(bars, completion_rates):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{rate:.1f}%', ha='left', va='center', fontweight='bold')

        # 2. Survey wave coverage
        ax2 = axes[0, 1]

        if 'survey_year' in gcro_data.columns:
            wave_counts = gcro_data['survey_year'].value_counts().sort_index()

            bars = ax2.bar(wave_counts.index, wave_counts.values,
                          color=self.colors['gcro'], alpha=0.8)
            ax2.set_xlabel('Survey Year')
            ax2.set_ylabel('Number of Records')
            ax2.set_title('GCRO Survey Wave Distribution', fontweight='bold')

            # Add count labels
            for bar, count in zip(bars, wave_counts.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 500,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')

        # 3. Selected variables rationale (focus on top 4)
        ax3 = axes[1, 0]
        ax3.axis('off')

        # Select top 4 variables based on completeness and relevance
        selected_variables = {
            'age': 'Continuous demographic variable, high completion, strong health predictor',
            'sex': 'Binary demographic variable, essential for health analysis stratification',
            'education': 'Ordinal socioeconomic indicator, linked to health behaviors and access',
            'income': 'Ordinal socioeconomic indicator, directly linked to heat vulnerability'
        }

        rationale_text = "SELECTED GCRO VARIABLES (Top 4):\n\n"
        for i, (var, rationale) in enumerate(selected_variables.items(), 1):
            completion = variable_completeness.get(var, 0)
            rationale_text += f"{i}. {var.upper()} ({completion:.1f}% complete)\n"
            rationale_text += f"   {rationale}\n\n"

        rationale_text += "SELECTION CRITERIA:\n"
        rationale_text += "• Data completeness (>80% preferred)\n"
        rationale_text += "• Theoretical relevance to heat vulnerability\n"
        rationale_text += "• Variable type diversity (continuous + categorical)\n"
        rationale_text += "• Prior literature support for climate-health associations"

        ax3.text(0.05, 0.95, rationale_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=FONT_SIZE - 1,
                bbox=dict(boxstyle='round,pad=1', facecolor=self.colors['gcro'], alpha=0.1))

        # 4. Variable distribution example (Age)
        ax4 = axes[1, 1]

        if 'age' in gcro_data.columns:
            age_data = pd.to_numeric(gcro_data['age'], errors='coerce').dropna()

            if len(age_data) > 0:
                ax4.hist(age_data, bins=30, color=self.colors['gcro'], alpha=0.7, edgecolor='black')
                ax4.set_xlabel('Age (years)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Age Distribution (GCRO)', fontweight='bold')

                # Add statistics
                mean_age = age_data.mean()
                median_age = age_data.median()
                ax4.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f}')
                ax4.axvline(median_age, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f}')
                ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'gcro_variable_analysis.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'gcro_variable_analysis.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved GCRO variable analysis to {self.output_dir}")

def main():
    """Create all Step 1 and Step 2 visualizations."""
    print("\n🎨 Creating publication-quality visualizations...")

    # Initialize components
    loader = HeatDataLoader(data_dir="../data")
    visualizer = HeatVisualizationCreator(output_dir="../figures")

    # Load data and validate
    print("📊 Loading and validating data...")
    df = loader.load_master_dataset(sample_fraction=1.0)
    validation_report = loader.validate_data_structure()
    clean_df = loader.get_clean_dataset()

    # Create visualizations
    print("🎯 Creating dataset overview...")
    visualizer.create_dataset_overview(validation_report, clean_df)

    print("🌡️ Analyzing climate data...")
    visualizer.create_climate_data_analysis(clean_df)

    print("📈 Analyzing GCRO variables...")
    visualizer.create_gcro_variable_analysis(clean_df)

    print(f"\n✅ All visualizations saved to ../figures/")
    print("📁 Generated files:")
    print("   • dataset_overview.svg/png")
    print("   • climate_data_analysis.svg/png")
    print("   • gcro_variable_analysis.svg/png")

if __name__ == "__main__":
    main()