#!/usr/bin/env python3
"""
HEAT Pipeline Preprocessing Visualization Creator
===============================================

Creates detailed visualizations for the preprocessing and feature engineering steps.

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
spec1 = importlib.util.spec_from_file_location("data_loader", "01_data_loader.py")
data_loader_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(data_loader_module)
HeatDataLoader = data_loader_module.HeatDataLoader

spec2 = importlib.util.spec_from_file_location("data_preprocessor", "02_data_preprocessor.py")
preprocessor_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocessor_module)
HeatDataPreprocessor = preprocessor_module.HeatDataPreprocessor

# Configure plotting
plt.style.use('default')
sns.set_palette("Set2")
warnings.filterwarnings('ignore')

# Publication-quality styling
FIGSIZE_LARGE = (16, 12)
FIGSIZE_MEDIUM = (12, 8)
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

class PreprocessingVisualizationCreator:
    """Creates visualizations for preprocessing and feature engineering."""

    def __init__(self, output_dir: str = "../figures"):
        """Initialize the visualization creator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F7DC6F',
            'neutral': '#7D8491',
            'climate': '#F18F01',
            'features': '#2E86AB',
            'missing': '#A23B72'
        }

    def create_feature_engineering_overview(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, report: dict):
        """Create overview of feature engineering process."""

        fig = plt.figure(figsize=FIGSIZE_LARGE)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Feature Engineering Overview: Climate-Health Analysis',
                    fontsize=TITLE_SIZE + 4, fontweight='bold', y=0.95)

        # 1. Before vs After comparison
        ax1 = fig.add_subplot(gs[0, :])

        categories = ['Original Features', 'Engineered Features']
        counts = [original_df.shape[1], processed_df.shape[1]]
        colors = [self.colors['neutral'], self.colors['features']]

        bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Feature Engineering Impact', fontweight='bold')

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=FONT_SIZE + 2)

        # Add improvement annotation
        improvement = counts[1] - counts[0]
        ax1.text(0.5, max(counts) * 0.8, f'+{improvement} features\nengineered',
                ha='center', va='center', fontsize=FONT_SIZE + 1, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['accent'], alpha=0.3))

        # 2. Feature category breakdown
        ax2 = fig.add_subplot(gs[1, 0])

        if 'feature_categories' in report:
            categories_dict = report['feature_categories']

            # Focus on key categories with content
            key_categories = {k: len(v) for k, v in categories_dict.items()
                            if v and k not in ['identifiers', 'exclude']}

            if key_categories:
                cat_names = list(key_categories.keys())
                cat_counts = list(key_categories.values())

                # Create pie chart
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_names)))
                wedges, texts, autotexts = ax2.pie(cat_counts, labels=cat_names, autopct='%1.1f%%',
                                                 colors=colors_pie, startangle=90)

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

                ax2.set_title('Feature Categories Distribution', fontweight='bold')

        # 3. Missing value handling summary
        ax3 = fig.add_subplot(gs[1, 1])

        # Calculate missing value statistics
        original_missing = original_df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        total_cells_original = original_df.size
        total_cells_processed = processed_df.size

        missing_stats = {
            'Original': (original_missing / total_cells_original) * 100,
            'After Processing': (processed_missing / total_cells_processed) * 100
        }

        stages = list(missing_stats.keys())
        percentages = list(missing_stats.values())
        colors_missing = [self.colors['missing'], self.colors['success']]

        bars = ax3.bar(stages, percentages, color=colors_missing, alpha=0.8)
        ax3.set_ylabel('Missing Data (%)')
        ax3.set_title('Missing Value Reduction', fontweight='bold')

        # Add value labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Target variables summary
        ax4 = fig.add_subplot(gs[2, :])

        if 'target_variables' in report:
            target_vars = report['target_variables']

            # Analyze target variable availability
            target_stats = []
            for target in target_vars:
                if target in processed_df.columns:
                    non_null = processed_df[target].notna().sum()
                    total = len(processed_df)
                    completion = (non_null / total) * 100

                    # Get basic statistics
                    values = pd.to_numeric(processed_df[target], errors='coerce').dropna()
                    if len(values) > 0:
                        target_stats.append({
                            'Variable': target.split('(')[0].strip(),
                            'N': non_null,
                            'Completion %': f'{completion:.1f}%',
                            'Mean': f'{values.mean():.2f}',
                            'Std': f'{values.std():.2f}',
                            'Min': f'{values.min():.2f}',
                            'Max': f'{values.max():.2f}'
                        })

            if target_stats:
                # Create table
                df_table = pd.DataFrame(target_stats)

                ax4.axis('tight')
                ax4.axis('off')

                table = ax4.table(cellText=df_table.values,
                                colLabels=df_table.columns,
                                cellLoc='center',
                                loc='center')

                table.auto_set_font_size(False)
                table.set_fontsize(FONT_SIZE - 1)
                table.scale(1, 2)

                # Style the table
                for i in range(len(df_table) + 1):
                    for j in range(len(df_table.columns)):
                        cell = table[(i, j)]
                        if i == 0:  # Header
                            cell.set_facecolor(self.colors['primary'])
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

                ax4.set_title('Target Variables Summary Statistics', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_engineering_overview.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'feature_engineering_overview.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved feature engineering overview to {self.output_dir}")

    def create_climate_feature_engineering(self, original_df: pd.DataFrame, processed_df: pd.DataFrame):
        """Visualize climate feature engineering process."""

        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)
        fig.suptitle('Climate Feature Engineering Analysis', fontsize=TITLE_SIZE + 2, fontweight='bold')

        # Identify original and engineered climate features
        original_climate = [col for col in original_df.columns if 'era5' in col.lower()]
        processed_climate = [col for col in processed_df.columns if any(term in col.lower()
                                                                       for term in ['era5', 'temp', 'climate', 'heat'])]

        # 1. Original vs engineered climate features
        ax1 = axes[0, 0]

        climate_counts = {
            'Original ERA5': len(original_climate),
            'Total Climate Features': len(processed_climate)
        }

        bars = ax1.bar(climate_counts.keys(), climate_counts.values(),
                      color=[self.colors['neutral'], self.colors['climate']], alpha=0.8)
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Climate Feature Expansion', fontweight='bold')

        # Add value labels
        for bar, count in zip(bars, climate_counts.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

        # 2. Temperature data distribution (if available)
        ax2 = axes[0, 1]

        # Look for temperature mean column
        temp_col = None
        for col in processed_climate:
            if 'temp' in col.lower() and 'mean' in col.lower():
                temp_col = col
                break

        if temp_col and temp_col in processed_df.columns:
            temp_data = pd.to_numeric(processed_df[temp_col], errors='coerce').dropna()

            if len(temp_data) > 0:
                ax2.hist(temp_data, bins=30, color=self.colors['climate'], alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Temperature (°C)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Temperature Distribution', fontweight='bold')

                # Add statistics
                mean_temp = temp_data.mean()
                std_temp = temp_data.std()
                ax2.axvline(mean_temp, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_temp:.1f}°C')
                ax2.axvline(mean_temp + std_temp, color='orange', linestyle=':', linewidth=2,
                           label=f'+1σ: {mean_temp + std_temp:.1f}°C')
                ax2.axvline(mean_temp - std_temp, color='orange', linestyle=':', linewidth=2,
                           label=f'-1σ: {mean_temp - std_temp:.1f}°C')
                ax2.legend()

        # 3. Feature engineering techniques used
        ax3 = axes[1, 0]
        ax3.axis('off')

        engineering_techniques = [
            "✓ Temperature variability calculation",
            "✓ Multi-temporal window analysis (1d, 7d, 30d)",
            "✓ Extreme temperature day counting",
            "✓ Heat stress indicator creation",
            "✓ Temperature quintile categorization",
            "✓ Lag-based temporal features"
        ]

        techniques_text = "CLIMATE FEATURE ENGINEERING TECHNIQUES:\n\n"
        techniques_text += "\n".join(engineering_techniques)
        techniques_text += "\n\nRATIONALE:\n"
        techniques_text += "• Temperature variability is more predictive than mean temperature\n"
        techniques_text += "• Multi-lag analysis captures delayed health effects\n"
        techniques_text += "• Extreme event indicators capture threshold effects\n"
        techniques_text += "• Categorical features capture non-linear relationships"

        ax3.text(0.05, 0.95, techniques_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=FONT_SIZE,
                bbox=dict(boxstyle='round,pad=1', facecolor=self.colors['climate'], alpha=0.1))

        # 4. Climate data completeness after processing
        ax4 = axes[1, 1]

        # Check completeness of key climate variables
        climate_completeness = {}
        for col in processed_climate[:5]:  # Top 5 climate features
            if col in processed_df.columns:
                completion = (processed_df[col].notna().sum() / len(processed_df)) * 100
                climate_completeness[col.replace('era5_temp_', '').replace('era5_', '')] = completion

        if climate_completeness:
            names = list(climate_completeness.keys())
            values = list(climate_completeness.values())

            bars = ax4.barh(names, values, color=self.colors['climate'], alpha=0.8)
            ax4.set_xlabel('Completion Rate (%)')
            ax4.set_title('Climate Feature Completeness', fontweight='bold')

            # Add value labels
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax4.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'climate_feature_engineering.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'climate_feature_engineering.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved climate feature engineering to {self.output_dir}")

    def create_data_quality_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, report: dict):
        """Create comprehensive data quality summary."""

        fig = plt.figure(figsize=FIGSIZE_LARGE)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle('Data Quality and Preprocessing Summary',
                    fontsize=TITLE_SIZE + 4, fontweight='bold', y=0.95)

        # 1. Dataset size changes
        ax1 = fig.add_subplot(gs[0, 0])

        size_comparison = {
            'Records': [original_df.shape[0], processed_df.shape[0]],
            'Features': [original_df.shape[1], processed_df.shape[1]]
        }

        x = np.arange(2)
        width = 0.35

        ax1.bar(x - width/2, size_comparison['Records'], width, label='Records',
               color=self.colors['primary'], alpha=0.8)
        ax1.bar(x + width/2, size_comparison['Features'], width, label='Features',
               color=self.colors['accent'], alpha=0.8)

        ax1.set_xlabel('Processing Stage')
        ax1.set_ylabel('Count')
        ax1.set_title('Dataset Size Changes', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Original', 'Processed'])
        ax1.legend()

        # Add value labels
        for i, (records, features) in enumerate(zip(size_comparison['Records'], size_comparison['Features'])):
            ax1.text(i - width/2, records + 1000, f'{records:,}', ha='center', va='bottom',
                    fontweight='bold', rotation=90)
            ax1.text(i + width/2, features + 5, f'{features}', ha='center', va='bottom',
                    fontweight='bold')

        # 2. Missing data pattern
        ax2 = fig.add_subplot(gs[0, 1])

        # Calculate missing percentages by column type
        numeric_cols_orig = original_df.select_dtypes(include=[np.number]).columns
        numeric_cols_proc = processed_df.select_dtypes(include=[np.number]).columns

        missing_by_type = {
            'Numeric (Original)': (original_df[numeric_cols_orig].isnull().sum().sum() / original_df[numeric_cols_orig].size) * 100,
            'Numeric (Processed)': (processed_df[numeric_cols_proc].isnull().sum().sum() / processed_df[numeric_cols_proc].size) * 100
        }

        bars = ax2.bar(missing_by_type.keys(), missing_by_type.values(),
                      color=[self.colors['missing'], self.colors['success']], alpha=0.8)
        ax2.set_ylabel('Missing Data (%)')
        ax2.set_title('Missing Data Reduction', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, value in zip(bars, missing_by_type.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 3. Feature type distribution
        ax3 = fig.add_subplot(gs[0, 2])

        if 'feature_categories' in report:
            categories_dict = report['feature_categories']

            # Count non-empty categories
            category_counts = {k: len(v) for k, v in categories_dict.items()
                             if v and k not in ['identifiers', 'exclude']}

            if category_counts:
                # Create horizontal bar chart
                categories = list(category_counts.keys())
                counts = list(category_counts.values())

                bars = ax3.barh(categories, counts, color=self.colors['features'], alpha=0.8)
                ax3.set_xlabel('Number of Features')
                ax3.set_title('Feature Type Distribution', fontweight='bold')

                # Add value labels
                for bar, count in zip(bars, counts):
                    width = bar.get_width()
                    ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                            f'{count}', ha='left', va='center', fontweight='bold')

        # 4. Processing pipeline summary
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        pipeline_summary = f"""
PREPROCESSING PIPELINE SUMMARY

📊 DATASET SCALE:
   • Original: {original_df.shape[0]:,} records × {original_df.shape[1]} features
   • Processed: {processed_df.shape[0]:,} records × {processed_df.shape[1]} features
   • Memory usage: {processed_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

🔧 PREPROCESSING STEPS COMPLETED:
   • Data type standardization and cleaning
   • Climate feature engineering ({report.get('features_engineered', 0)} new features)
   • Missing value imputation (adaptive strategy)
   • Categorical variable encoding
   • Feature scaling and normalization

🎯 TARGET VARIABLES PREPARED:
   • {len(report.get('target_variables', []))} biomarkers ready for analysis
   • Focus on cardiovascular, metabolic, and immune function markers

⚠️ QUALITY ASSURANCE:
   • Data integrity validated at each step
   • Temporal ordering preserved for cross-validation
   • Geographic coordinates validated
   • Clinical range validation applied

📈 READY FOR MACHINE LEARNING:
   • Feature matrix prepared for Random Forest models
   • Temporal cross-validation structure maintained
   • SHAP explainability analysis enabled
        """

        ax4.text(0.05, 0.95, pipeline_summary, transform=ax4.transAxes,
                verticalalignment='top', fontsize=FONT_SIZE,
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_quality_summary.svg',
                   dpi=DPI, bbox_inches='tight', format='svg')
        plt.savefig(self.output_dir / 'data_quality_summary.png',
                   dpi=DPI, bbox_inches='tight', format='png')
        plt.close()

        print(f"✅ Saved data quality summary to {self.output_dir}")

def main():
    """Create all preprocessing visualizations."""
    print("\n🎨 Creating preprocessing visualizations...")

    # Load data and run preprocessing
    print("📊 Loading data...")
    loader = HeatDataLoader(data_dir="../data")
    df_original = loader.load_master_dataset(sample_fraction=1.0)
    df_clean = loader.get_clean_dataset()

    print("🔧 Running preprocessing...")
    preprocessor = HeatDataPreprocessor()
    df_processed, preprocessing_report = preprocessor.preprocess_complete_pipeline(df_clean)

    # Create visualizations
    visualizer = PreprocessingVisualizationCreator(output_dir="../figures")

    print("📈 Creating feature engineering overview...")
    visualizer.create_feature_engineering_overview(df_original, df_processed, preprocessing_report)

    print("🌡️ Creating climate feature engineering analysis...")
    visualizer.create_climate_feature_engineering(df_original, df_processed)

    print("📊 Creating data quality summary...")
    visualizer.create_data_quality_summary(df_original, df_processed, preprocessing_report)

    print(f"\n✅ All preprocessing visualizations saved to ../figures/")
    print("📁 Generated files:")
    print("   • feature_engineering_overview.svg/png")
    print("   • climate_feature_engineering.svg/png")
    print("   • data_quality_summary.svg/png")

    return df_processed, preprocessing_report

if __name__ == "__main__":
    df_processed, report = main()