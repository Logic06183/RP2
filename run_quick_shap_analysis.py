#!/usr/bin/env python3
"""
Quick SHAP Analysis for Johannesburg Clinical Sites
==================================================
Optimized version to run within time constraints while maintaining quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

def run_quick_analysis():
    """Run optimized SHAP analysis."""
    print("=" * 70)
    print("🚀 QUICK JOHANNESBURG CLINICAL SITES SHAP ANALYSIS")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load data
    data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/ENHANCED_CLIMATE_BIOMARKER_DATASET.csv"
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
    except:
        print("❌ Could not load enhanced dataset")
        return
    
    # Filter to Johannesburg area (using clinical data subset)
    jhb_data = df[df['CD4 cell count (cells/µL)'].notna() | 
                  df['Hemoglobin (g/dL)'].notna() | 
                  df['systolic blood pressure'].notna()].copy()
    
    print(f"🌍 Clinical data subset: {len(jhb_data):,} records")
    
    # Quick analysis for 3 key biomarkers
    biomarkers = {
        'Blood_Pressure': 'systolic blood pressure',
        'Hemoglobin': 'Hemoglobin (g/dL)',
        'CD4_Count': 'CD4 cell count (cells/µL)'
    }
    
    results_summary = []
    
    for name, outcome in biomarkers.items():
        print(f"\n{'='*50}")
        print(f"🧪 Analyzing: {name}")
        print(f"{'='*50}")
        
        # Filter to biomarker records
        bio_data = jhb_data[jhb_data[outcome].notna()].copy()
        
        if len(bio_data) < 100:
            print(f"⚠️ Insufficient data: {len(bio_data)} samples")
            continue
        
        # Select predictors
        climate_vars = [col for col in bio_data.columns if col.startswith('climate_')][:10]
        demo_vars = ['Age (at enrolment)', 'Sex', 'Race']
        
        all_vars = []
        for var in climate_vars + demo_vars:
            if var in bio_data.columns and bio_data[var].notna().sum() > len(bio_data) * 0.5:
                all_vars.append(var)
        
        if len(all_vars) < 5:
            print(f"⚠️ Insufficient predictors: {len(all_vars)}")
            continue
        
        # Prepare data
        X = bio_data[all_vars].copy()
        y = bio_data[outcome].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('missing')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Remove any remaining NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"📊 Dataset: {len(X)} samples, {len(all_vars)} predictors")
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = xgb.XGBRegressor(
            n_estimators=100,  # Reduced for speed
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"📈 Performance: R² = {r2:.3f}, RMSE = {rmse:.2f}")
        
        # SHAP analysis
        print(f"🔍 Computing SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        # Categorize features
        climate_features = feature_importance[feature_importance['feature'].str.startswith('climate_')]
        demo_features = feature_importance[~feature_importance['feature'].str.startswith('climate_')]
        
        climate_total = climate_features['importance'].sum()
        demo_total = demo_features['importance'].sum()
        total = climate_total + demo_total
        
        if total > 0:
            climate_pct = (climate_total / total) * 100
            demo_pct = (demo_total / total) * 100
        else:
            climate_pct = demo_pct = 0
        
        print(f"📊 Pathway Contributions:")
        print(f"   🌡️ Climate: {climate_pct:.1f}%")
        print(f"   👥 Demographic: {demo_pct:.1f}%")
        
        print(f"🌡️ Top 5 Climate Features:")
        for idx, row in climate_features.head(5).iterrows():
            clean_name = row['feature'].replace('climate_', '').replace('_', ' ').title()
            print(f"   • {clean_name}: {row['importance']:.4f}")
        
        # Store results
        results_summary.append({
            'biomarker': name,
            'outcome': outcome,
            'samples': len(X),
            'r2': r2,
            'rmse': rmse,
            'climate_pct': climate_pct,
            'demo_pct': demo_pct,
            'top_climate': climate_features.head(1)['feature'].values[0] if len(climate_features) > 0 else 'None',
            'top_demo': demo_features.head(1)['feature'].values[0] if len(demo_features) > 0 else 'None'
        })
        
        # Create quick visualization
        create_quick_visualization(name, outcome, feature_importance, shap_values, 
                                 X_test[:100], r2, climate_pct)
    
    # Generate summary report
    generate_summary_report(results_summary)
    
    print(f"\n✅ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")

def create_quick_visualization(name, outcome, feature_importance, shap_values, X_test, r2, climate_pct):
    """Create quick visualization for each biomarker."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Top 10 Features
    ax = axes[0, 0]
    top_features = feature_importance.head(10)
    colors = ['#FF6B6B' if f.startswith('climate_') else '#45B7D1' 
              for f in top_features['feature']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    labels = [f.replace('climate_', '').replace('_', ' ')[:20] 
              for f in top_features['feature']]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('SHAP Importance')
    ax.set_title('Top 10 Features', fontweight='bold')
    
    # 2. Pathway Contributions
    ax = axes[0, 1]
    sizes = [climate_pct, 100-climate_pct]
    labels = ['Climate/Heat', 'Demographic']
    colors = ['#FF6B6B', '#45B7D1']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Pathway Contributions', fontweight='bold')
    
    # 3. SHAP Summary
    ax = axes[1, 0]
    top_indices = feature_importance.head(8).index
    for i, idx in enumerate(top_indices):
        col_idx = list(X_test.columns).index(feature_importance.loc[idx, 'feature'])
        y_pos = [i] * len(shap_values[:, col_idx])
        ax.scatter(shap_values[:, col_idx], y_pos, alpha=0.6, s=20)
    
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels([feature_importance.loc[idx, 'feature'][:15] 
                        for idx in top_indices], fontsize=9)
    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Value Distribution', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Model Info
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""Model Performance
    
Biomarker: {outcome}
R² Score: {r2:.3f}
Samples: {len(X_test)}

Climate Contribution: {climate_pct:.1f}%
Demographic: {100-climate_pct:.1f}%

Top Climate Feature:
{feature_importance[feature_importance['feature'].str.startswith('climate_')].head(1)['feature'].values[0] if any(feature_importance['feature'].str.startswith('climate_')) else 'None'}
"""
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=11, va='center')
    
    plt.suptitle(f'{name} SHAP Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save as SVG
    filename = f"quick_shap_{name.lower()}.svg"
    filepath = Path("figures") / filename
    plt.savefig(filepath, format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved visualization: {filepath}")

def generate_summary_report(results_summary):
    """Generate summary report."""
    print("\n" + "="*70)
    print("📋 ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    if not results_summary:
        print("❌ No results to summarize")
        return
    
    df_results = pd.DataFrame(results_summary)
    
    print("\n📊 BIOMARKER ANALYSIS RESULTS")
    print("-"*40)
    for _, row in df_results.iterrows():
        print(f"\n{row['biomarker']}:")
        print(f"  • Samples: {row['samples']:,}")
        print(f"  • R² Score: {row['r2']:.3f}")
        print(f"  • Climate Contribution: {row['climate_pct']:.1f}%")
        print(f"  • Top Climate Variable: {row['top_climate'].replace('climate_', '').replace('_', ' ')}")
    
    print("\n📈 OVERALL INSIGHTS")
    print("-"*40)
    print(f"• Average R²: {df_results['r2'].mean():.3f}")
    print(f"• Average Climate Contribution: {df_results['climate_pct'].mean():.1f}%")
    print(f"• Total Samples Analyzed: {df_results['samples'].sum():,}")
    
    # Save summary to CSV
    summary_path = Path("figures") / "analysis_summary.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"\n✅ Summary saved to: {summary_path}")

if __name__ == "__main__":
    run_quick_analysis()