#!/usr/bin/env python3
"""
Fixed Socioeconomic SHAP Analysis
=================================
Ensures both climate AND socioeconomic variables appear in SHAP analysis
by properly integrating GCRO survey data with clinical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import resample
import xgboost as xgb
import shap
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure matplotlib for beautiful styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12

def run_fixed_socioeconomic_analysis():
    """Run analysis ensuring socioeconomic variables are included."""
    print("=" * 80)
    print("🔧 FIXED SOCIOECONOMIC + CLIMATE SHAP ANALYSIS")
    print("=" * 80)
    print("Ensuring both climate AND socioeconomic variables appear in results")
    print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Load enhanced dataset
    data_path = "/home/cparker/HEAT_RESEARCH_PROJECTS/heat_analysis_projects/heat_analysis_optimized/data/ENHANCED_CLIMATE_BIOMARKER_DATASET.csv"
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
    except:
        print("❌ Could not load enhanced dataset")
        return
    
    # Strategy: Include ALL Johannesburg-area data (not just clinical sites)
    print(f"\n📊 INVESTIGATING SOCIOECONOMIC DATA COVERAGE")
    print("-" * 50)
    
    # Check socioeconomic variable coverage across full dataset
    socio_vars = ['employment', 'education', 'income', 'municipality', 'ward']
    available_socio = []
    
    for var in socio_vars:
        if var in df.columns:
            coverage = df[var].notna().sum()
            coverage_pct = (coverage / len(df)) * 100
            print(f"📊 {var}: {coverage:,} records ({coverage_pct:.1f}%)")
            if coverage > 1000:  # Reasonable threshold
                available_socio.append(var)
    
    print(f"\n✅ Available socioeconomic variables: {available_socio}")
    
    # Create INCLUSIVE dataset that preserves socioeconomic data
    print(f"\n🔄 CREATING INCLUSIVE DATASET")
    print("-" * 35)
    
    # Strategy: Separate data sources then combine strategically
    data_sources = {}
    
    # 1. Clinical biomarker data (RP2)
    clinical_mask = (df['CD4 cell count (cells/µL)'].notna() | 
                    df['Hemoglobin (g/dL)'].notna() | 
                    df['Creatinine (mg/dL)'].notna())
    data_sources['clinical'] = df[clinical_mask].copy()
    
    # 2. Blood pressure data  
    bp_mask = (df['systolic blood pressure'].notna() | 
               df['diastolic blood pressure'].notna())
    data_sources['blood_pressure'] = df[bp_mask].copy()
    
    # 3. GCRO socioeconomic survey data (with any biomarkers)
    socio_mask = df[available_socio].notna().any(axis=1) if available_socio else pd.Series([False]*len(df))
    gcro_data = df[socio_mask].copy()
    
    # Keep GCRO records that also have biomarkers
    gcro_with_bio = gcro_data[
        gcro_data['systolic blood pressure'].notna() |
        gcro_data['diastolic blood pressure'].notna() |
        gcro_data['CD4 cell count (cells/µL)'].notna() |
        gcro_data['Hemoglobin (g/dL)'].notna()
    ].copy()
    
    data_sources['socioeconomic'] = gcro_with_bio
    
    print(f"Data source breakdown:")
    for source, data in data_sources.items():
        print(f"   {source}: {len(data):,} records")
    
    # Create balanced combined dataset
    balanced_dfs = []
    target_size = 2000  # Reasonable size per source
    
    for source, data in data_sources.items():
        if len(data) > 0:
            if len(data) > target_size:
                balanced = resample(data, n_samples=target_size, random_state=42, replace=False)
                print(f"   ↓ Downsampled {source}: {len(data):,} → {len(balanced):,}")
            else:
                balanced = data
                print(f"   ✓ Kept {source}: {len(balanced):,}")
            balanced_dfs.append(balanced)
    
    # Combine and remove duplicates
    combined_data = pd.concat(balanced_dfs, ignore_index=True)
    combined_data = combined_data.drop_duplicates()
    
    print(f"\n✅ Final combined dataset: {len(combined_data):,} records")
    
    # Verify socioeconomic coverage in final dataset
    print(f"\n🏢 Socioeconomic coverage in final dataset:")
    for var in available_socio:
        coverage = combined_data[var].notna().sum()
        coverage_pct = (coverage / len(combined_data)) * 100
        print(f"   • {var}: {coverage:,} records ({coverage_pct:.1f}%)")
    
    # Run enhanced analysis for key biomarkers
    biomarkers = {
        'Systolic_BP': 'systolic blood pressure',
        'Hemoglobin': 'Hemoglobin (g/dL)',  
        'CD4_Count': 'CD4 cell count (cells/µL)'
    }
    
    results_summary = []
    
    for name, outcome in biomarkers.items():
        print(f"\n{'='*60}")
        print(f"🧪 ENHANCED ANALYSIS: {name}")
        print(f"{'='*60}")
        
        result = run_enhanced_biomarker_analysis(combined_data, name, outcome, available_socio)
        if result:
            results_summary.append(result)
            create_enhanced_visualization_with_socio(result, name, outcome)
    
    # Generate final summary
    generate_enhanced_summary(results_summary)
    
    print(f"\n✅ Fixed analysis completed at {datetime.now().strftime('%H:%M:%S')}")

def run_enhanced_biomarker_analysis(data, name, outcome, available_socio):
    """Run enhanced analysis ensuring socioeconomic variables are included."""
    # Filter to biomarker records
    bio_data = data[data[outcome].notna()].copy()
    
    if len(bio_data) < 100:
        print(f"⚠️ Insufficient data: {len(bio_data)} samples")
        return None
    
    # Select comprehensive predictors
    climate_vars = [col for col in bio_data.columns if col.startswith('climate_')][:12]
    demo_vars = ['Age (at enrolment)', 'Sex', 'Race']
    
    all_predictors = []
    
    # Add climate variables
    for var in climate_vars:
        if var in bio_data.columns and bio_data[var].notna().sum() > len(bio_data) * 0.7:
            all_predictors.append(var)
    
    # Add socioeconomic variables - FORCE INCLUSION even with lower coverage
    socio_included = []
    for var in available_socio:
        if var in bio_data.columns and bio_data[var].notna().sum() > 50:  # Lower threshold
            all_predictors.append(var)
            socio_included.append(var)
    
    # Add demographic variables
    for var in demo_vars:
        if var in bio_data.columns and bio_data[var].notna().sum() > len(bio_data) * 0.3:
            all_predictors.append(var)
    
    print(f"📊 Predictors selected:")
    print(f"   • Climate: {len([p for p in all_predictors if p.startswith('climate_')])}")
    print(f"   • Socioeconomic: {len(socio_included)} - {socio_included}")
    print(f"   • Demographic: {len([p for p in all_predictors if p in demo_vars])}")
    
    if len(all_predictors) < 5:
        print(f"⚠️ Insufficient predictors: {len(all_predictors)}")
        return None
    
    # Prepare data with enhanced preprocessing for socioeconomic variables
    X = bio_data[all_predictors].copy()
    y = bio_data[outcome].copy()
    
    # Handle missing values with special attention to socioeconomic vars
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('missing')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        else:
            if col in socio_included:
                # For socioeconomic vars, use mode or specific imputation
                if X[col].nunique() < 10:  # Categorical numeric
                    X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
                else:
                    X[col] = X[col].fillna(X[col].median())
            else:
                # Standard median imputation for other vars
                X[col] = X[col].fillna(X[col].median())
    
    # Remove any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"📊 Final dataset: {len(X)} samples, {len(all_predictors)} predictors")
    
    if len(X) < 50:
        print(f"⚠️ Insufficient final samples: {len(X)}")
        return None
    
    # Train enhanced model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
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
    shap_values = explainer.shap_values(X_test)
    
    # Feature importance with pathway categorization
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    # Categorize features
    climate_features = feature_importance[feature_importance['feature'].str.startswith('climate_')]
    socio_features = feature_importance[feature_importance['feature'].isin(available_socio)]
    demo_features = feature_importance[
        ~feature_importance['feature'].str.startswith('climate_') & 
        ~feature_importance['feature'].isin(available_socio)
    ]
    
    # Calculate contributions
    climate_total = climate_features['importance'].sum()
    socio_total = socio_features['importance'].sum()
    demo_total = demo_features['importance'].sum()
    total = climate_total + socio_total + demo_total
    
    if total > 0:
        climate_pct = (climate_total / total) * 100
        socio_pct = (socio_total / total) * 100
        demo_pct = (demo_total / total) * 100
    else:
        climate_pct = socio_pct = demo_pct = 0
    
    print(f"📊 Pathway Contributions:")
    print(f"   🌡️ Climate: {climate_pct:.1f}%")
    print(f"   🏢 Socioeconomic: {socio_pct:.1f}%")
    print(f"   👥 Demographic: {demo_pct:.1f}%")
    
    # Display top effects
    if len(climate_features) > 0:
        print(f"🌡️ Top 5 Climate Features:")
        for _, row in climate_features.head(5).iterrows():
            clean_name = row['feature'].replace('climate_', '').replace('_', ' ').title()
            print(f"   • {clean_name}: {row['importance']:.4f}")
    
    if len(socio_features) > 0:
        print(f"🏢 Top Socioeconomic Features:")
        for _, row in socio_features.head().iterrows():
            print(f"   • {row['feature'].title()}: {row['importance']:.4f}")
    else:
        print(f"⚠️ No socioeconomic features found in model")
    
    return {
        'name': name,
        'outcome': outcome,
        'samples': len(X),
        'r2': r2,
        'rmse': rmse,
        'climate_pct': climate_pct,
        'socio_pct': socio_pct,
        'demo_pct': demo_pct,
        'climate_features': climate_features,
        'socio_features': socio_features,
        'demo_features': demo_features,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'X_test': X_test
    }

def create_enhanced_visualization_with_socio(result, name, outcome):
    """Create enhanced visualization showing both climate AND socioeconomic effects."""
    print(f"   🎨 Creating enhanced visualization with socioeconomic variables...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Top 8 Climate Effects (Top Left)
    ax = axes[0, 0]
    climate_features = result['climate_features']
    
    if len(climate_features) > 0:
        top_climate = climate_features.head(8)
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_climate)))
        
        bars = ax.barh(range(len(top_climate)), top_climate['importance'], color=colors)
        ax.set_yticks(range(len(top_climate)))
        labels = [f.replace('climate_', '').replace('_', ' ').title()[:18] 
                 for f in top_climate['feature']]
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('SHAP Importance', fontsize=10)
        ax.set_title('🌡️ Top Climate/Heat Effects', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add values
        for bar, imp in zip(bars, top_climate['importance']):
            ax.text(bar.get_width() + max(top_climate['importance'])*0.01, 
                   bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No Climate Effects', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
    
    # 2. Socioeconomic Effects (Top Center) - KEY FIX
    ax = axes[0, 1]
    socio_features = result['socio_features']
    
    if len(socio_features) > 0:
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(socio_features)))
        
        bars = ax.barh(range(len(socio_features)), socio_features['importance'], color=colors)
        ax.set_yticks(range(len(socio_features)))
        labels = [f.replace('_', ' ').title() for f in socio_features['feature']]
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('SHAP Importance', fontsize=10)
        ax.set_title('🏢 Socioeconomic Effects', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add values
        for bar, imp in zip(bars, socio_features['importance']):
            ax.text(bar.get_width() + max(socio_features['importance'])*0.01,
                   bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax.text(0.5, 0.5, '⚠️ No Socioeconomic\nEffects Found', ha='center', va='center',
               transform=ax.transAxes, fontsize=11, color='red')
        ax.set_title('🏢 Socioeconomic Effects', fontweight='bold', fontsize=11)
    
    # 3. Enhanced Pathway Contributions (Top Right)
    ax = axes[0, 2]
    if result['climate_pct'] + result['socio_pct'] + result['demo_pct'] > 0:
        sizes = [result['climate_pct'], result['socio_pct'], result['demo_pct']]
        labels = ['Climate/Heat', 'Socioeconomic', 'Demographic']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        ax.set_title('📊 Enhanced Pathway\nContributions', fontweight='bold', fontsize=11)
    
    # 4. SHAP Summary (Bottom Left)
    ax = axes[1, 0]
    top_features = result['feature_importance'].head(8)
    
    for i, (_, row) in enumerate(top_features.iterrows()):
        feature_name = row['feature']
        if feature_name in result['X_test'].columns:
            col_idx = list(result['X_test'].columns).index(feature_name)
            y_pos = [i] * len(result['shap_values'][:, col_idx])
            
            # Color by pathway type
            if feature_name.startswith('climate_'):
                color = '#FF6B6B'
            elif feature_name in ['employment', 'education', 'income', 'municipality', 'ward']:
                color = '#4ECDC4'
            else:
                color = '#45B7D1'
                
            ax.scatter(result['shap_values'][:, col_idx], y_pos, alpha=0.6, s=15, c=color)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([f[:15] + '...' if len(f) > 15 else f 
                        for f in top_features['feature']], fontsize=9)
    ax.set_xlabel('SHAP Value (impact on prediction)', fontsize=10)
    ax.set_title('🔍 SHAP Value Distribution', fontweight='bold', fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 5. Model Performance (Bottom Center)
    ax = axes[1, 1]
    ax.axis('off')
    
    performance_text = [
        f"📊 Model Performance",
        "",
        f"Biomarker: {outcome}",
        f"R² Score: {result['r2']:.3f}",
        f"RMSE: {result['rmse']:.2f}",
        f"Samples: {result['samples']:,}",
        "",
        f"Pathway Breakdown:",
        f"🌡️ Climate: {result['climate_pct']:.1f}%",
        f"🏢 Socioeconomic: {result['socio_pct']:.1f}%",
        f"👥 Demographic: {result['demo_pct']:.1f}%",
        "",
        f"Variables:",
        f"• Climate: {len(result['climate_features'])}",
        f"• Socioeconomic: {len(result['socio_features'])}",
        f"• Demographic: {len(result['demo_features'])}"
    ]
    
    for i, text in enumerate(performance_text):
        weight = 'bold' if text.endswith(':') else 'normal'
        fontsize = 11 if i == 0 else 10
        ax.text(0.05, 0.95 - i*0.058, text, transform=ax.transAxes,
               fontsize=fontsize, weight=weight, va='top')
    
    # 6. Top 10 Overall Features (Bottom Right)
    ax = axes[1, 2]
    top_overall = result['feature_importance'].head(10)
    
    # Color by pathway
    colors = []
    for feature in top_overall['feature']:
        if feature.startswith('climate_'):
            colors.append('#FF6B6B')
        elif feature in ['employment', 'education', 'income', 'municipality', 'ward']:
            colors.append('#4ECDC4')
        else:
            colors.append('#45B7D1')
    
    bars = ax.barh(range(len(top_overall)), top_overall['importance'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_overall)))
    ax.set_yticklabels([f[:12] + '...' if len(f) > 12 else f 
                        for f in top_overall['feature']], fontsize=9)
    ax.set_xlabel('SHAP Importance', fontsize=10)
    ax.set_title('🏆 Top 10 Overall Predictors', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Climate'),
        Patch(facecolor='#4ECDC4', label='Socioeconomic'),
        Patch(facecolor='#45B7D1', label='Demographic')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.suptitle(f'Enhanced {name} Analysis - Climate + Socioeconomic SHAP', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save as SVG
    filename = f"fixed_socio_shap_{name.lower()}.svg"
    filepath = Path("figures") / filename
    plt.savefig(filepath, format='svg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Enhanced visualization saved: {filepath}")

def generate_enhanced_summary(results_summary):
    """Generate enhanced summary including socioeconomic results."""
    print(f"\n{'='*80}")
    print("📋 ENHANCED ANALYSIS SUMMARY - Climate + Socioeconomic")
    print("="*80)
    
    if not results_summary:
        print("❌ No results to summarize")
        return
    
    for result in results_summary:
        print(f"\n🧪 {result['name']}:")
        print(f"   • Samples: {result['samples']:,}")
        print(f"   • R² Score: {result['r2']:.3f}")
        print(f"   • Climate Contribution: {result['climate_pct']:.1f}%")
        print(f"   • Socioeconomic Contribution: {result['socio_pct']:.1f}%")  # KEY METRIC
        print(f"   • Demographic Contribution: {result['demo_pct']:.1f}%")
        
        if len(result['socio_features']) > 0:
            print(f"   • Top Socioeconomic Variable: {result['socio_features'].iloc[0]['feature']}")
        else:
            print(f"   • ⚠️ No socioeconomic variables detected")
    
    # Overall averages
    avg_climate = np.mean([r['climate_pct'] for r in results_summary])
    avg_socio = np.mean([r['socio_pct'] for r in results_summary])
    avg_demo = np.mean([r['demo_pct'] for r in results_summary])
    
    print(f"\n📊 OVERALL AVERAGES:")
    print(f"   🌡️ Climate: {avg_climate:.1f}%")
    print(f"   🏢 Socioeconomic: {avg_socio:.1f}%")
    print(f"   👥 Demographic: {avg_demo:.1f}%")
    
    if avg_socio > 5:
        print(f"\n✅ SUCCESS: Socioeconomic variables now appearing in SHAP analysis!")
    else:
        print(f"\n⚠️  Socioeconomic variables still low - may need broader geographic inclusion")

if __name__ == "__main__":
    run_fixed_socioeconomic_analysis()