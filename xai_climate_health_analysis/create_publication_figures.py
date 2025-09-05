#!/usr/bin/env python3
"""
Publication-Quality Figure Generation
Author: Craig Parker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PublicationFigures:
    def __init__(self):
        self.results_dir = Path("rigorous_results")
        
    def load_results(self):
        with open(self.results_dir / 'xai_results.json') as f:
            return json.load(f)
    
    def create_main_results_figure(self, results):
        """Figure 1: Main XAI analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: R² scores
        biomarkers = [r['biomarker'] for r in results if r and 'biomarker' in r]
        r2_scores = [r['r2_score'] for r in results if r and 'biomarker' in r]
        sample_sizes = [r['n_samples'] for r in results if r and 'biomarker' in r]
        
        # Clean biomarker names for display
        clean_names = []
        for b in biomarkers:
            clean = b.replace('FASTING ', '').replace('systolic blood pressure', 'Systolic BP')
            clean = clean.replace('CREATININE', 'Creatinine').replace('HEMOGLOBIN', 'Hemoglobin')
            clean_names.append(clean)
        
        bars = ax1.bar(range(len(clean_names)), r2_scores, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Biomarker', fontsize=12)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_title('A. Climate-Health XAI Model Performance', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(clean_names)))
        ax1.set_xticklabels(clean_names, rotation=45, ha='right')
        ax1.set_ylim(0, max(r2_scores) * 1.1)
        
        # Add R² values on bars
        for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Panel B: Sample sizes
        bars2 = ax2.bar(range(len(clean_names)), sample_sizes, color='coral', alpha=0.8)
        ax2.set_xlabel('Biomarker', fontsize=12)
        ax2.set_ylabel('Sample Size (n)', fontsize=12)
        ax2.set_title('B. Analysis Sample Sizes', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(clean_names)))
        ax2.set_xticklabels(clean_names, rotation=45, ha='right')
        
        # Add sample sizes on bars
        for i, (bar, n) in enumerate(zip(bars2, sample_sizes)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{n:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure_1_main_results.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.results_dir / 'figure_1_main_results.svg'
    
    def create_feature_importance_figure(self, results):
        """Figure 2: Climate feature importance"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, result in enumerate(results[:9]):  # Top 9 biomarkers
            if result and 'feature_importance' in result:
                features = list(result['feature_importance'].keys())
                importances = list(result['feature_importance'].values())
                biomarker = result['biomarker'].replace('FASTING ', '').replace('systolic blood pressure', 'Systolic BP')
                
                ax = axes[i]
                bars = ax.bar(range(len(features)), importances, color='forestgreen', alpha=0.7)
                ax.set_title(f'{biomarker}\n(R² = {result["r2_score"]:.3f})', fontsize=10, fontweight='bold')
                ax.set_xticks(range(len(features)))
                ax.set_xticklabels([f.replace('temp_', '').replace('_', ' ').title() for f in features], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Importance', fontsize=9)
                
                # Add importance values
                for bar, imp in zip(bars, importances):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{imp:.2f}', ha='center', va='bottom', fontsize=7)
        
        plt.suptitle('Climate Feature Importance Across Biomarkers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'figure_2_feature_importance.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.results_dir / 'figure_2_feature_importance.svg'
    
    def create_performance_summary_table(self, results):
        """Table 1: Performance summary"""
        table_data = []
        for result in results:
            if result and 'biomarker' in result:
                table_data.append({
                    'Biomarker': result['biomarker'].replace('FASTING ', ''),
                    'Sample Size': f"{result['n_samples']:,}",
                    'R² Score': f"{result['r2_score']:.4f}",
                    'Model Performance': 'Excellent' if result['r2_score'] > 0.6 else 'Good' if result['r2_score'] > 0.4 else 'Moderate'
                })
        
        df = pd.DataFrame(table_data)
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows based on performance
        for i in range(1, len(df) + 1):
            r2_val = float(df.iloc[i-1]['R² Score'])
            if r2_val > 0.6:
                color = '#E2F0D9'  # Light green
            elif r2_val > 0.4:
                color = '#FFF2CC'  # Light yellow
            else:
                color = '#FCE4D6'  # Light orange
                
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Table 1: Climate-Health XAI Analysis Results Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.results_dir / 'table_1_summary.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.results_dir / 'table_1_summary.svg'
    
    def generate_all_figures(self):
        """Generate all publication figures"""
        results = self.load_results()
        
        print("Generating publication figures...")
        
        fig1 = self.create_main_results_figure(results)
        print(f"✓ Created Figure 1: {fig1}")
        
        fig2 = self.create_feature_importance_figure(results)
        print(f"✓ Created Figure 2: {fig2}")
        
        table1 = self.create_performance_summary_table(results)
        print(f"✓ Created Table 1: {table1}")
        
        return [fig1, fig2, table1]

if __name__ == "__main__":
    generator = PublicationFigures()
    figures = generator.generate_all_figures()
    print(f"\nAll figures generated successfully in rigorous_results/")