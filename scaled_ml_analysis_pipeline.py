#!/usr/bin/env python3
"""
Scaled Machine Learning Analysis Pipeline
Designed to handle 128k+ integrated records efficiently

This pipeline implements advanced ML techniques with proper bias control,
cross-validation, and explainable AI for the complete HEAT Center dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ScaledMLPipeline:
    """
    Scaled ML pipeline for massive integrated dataset analysis.
    Handles 128k+ records efficiently with batching and sampling strategies.
    """
    
    def __init__(self, data_path="data/MASTER_INTEGRATED_DATASET.csv"):
        self.data_path = data_path
        self.data = None
        self.preprocessed_data = None
        self.models = {}
        self.results = {}
        self.shap_values = {}
        
    def load_and_sample_data(self, sample_fraction=1.0):
        """Load data with optional sampling for memory efficiency."""
        print("=" * 60)
        print("📊 LOADING MASTER INTEGRATED DATASET")
        print("=" * 60)
        
        # Load data
        print(f"Loading from: {self.data_path}")
        self.data = pd.read_csv(self.data_path, low_memory=False)
        
        print(f"✅ Loaded: {len(self.data):,} records, {len(self.data.columns):,} columns")
        
        # Data source breakdown
        if 'data_source' in self.data.columns:
            source_counts = self.data['data_source'].value_counts()
            print("\n📈 Data sources:")
            for source, count in source_counts.items():
                print(f"   • {source}: {count:,} records")
        
        # Sample if needed for memory efficiency
        if sample_fraction < 1.0:
            n_samples = int(len(self.data) * sample_fraction)
            self.data = self.data.sample(n=n_samples, random_state=42)
            print(f"\n📊 Sampled to {len(self.data):,} records ({sample_fraction*100:.0f}%)")
        
        return self.data
    
    def preprocess_data(self):
        """Comprehensive data preprocessing for ML."""
        print("\n" + "=" * 60)
        print("🔧 DATA PREPROCESSING")
        print("=" * 60)
        
        # Identify numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns
        id_cols = [col for col in numeric_cols if 'id' in col.lower() or 'index' in col.lower()]
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        print(f"📊 Numeric features: {len(numeric_cols)}")
        
        # Handle missing values
        print("\n🔄 Handling missing values...")
        imputer = SimpleImputer(strategy='median')
        
        if numeric_cols:
            # Select only columns that exist in the data
            existing_cols = [col for col in numeric_cols if col in self.data.columns]
            
            if existing_cols:
                imputed_data = imputer.fit_transform(self.data[existing_cols])
                self.preprocessed_data = pd.DataFrame(
                    imputed_data,
                    columns=existing_cols,
                    index=self.data.index
                )
            else:
                print("❌ No valid numeric columns found")
                self.preprocessed_data = pd.DataFrame()
            
            # Add categorical columns if needed
            if 'data_source' in self.data.columns:
                self.preprocessed_data['is_clinical'] = (self.data['data_source'] == 'RP2_Clinical').astype(int)
                self.preprocessed_data['is_gcro'] = (self.data['data_source'] == 'GCRO').astype(int)
            
            # Add temporal features if available
            if 'survey_year' in self.data.columns:
                self.preprocessed_data['survey_year'] = self.data['survey_year'].fillna(2015)
            
            print(f"✅ Preprocessed: {len(self.preprocessed_data):,} records, {len(self.preprocessed_data.columns):,} features")
        else:
            print("❌ No numeric columns found for preprocessing")
            self.preprocessed_data = pd.DataFrame()
        
        return self.preprocessed_data
    
    def create_analysis_targets(self):
        """Create meaningful analysis targets from available data."""
        print("\n" + "=" * 60)
        print("🎯 CREATING ANALYSIS TARGETS")
        print("=" * 60)
        
        targets = {}
        
        # Check for health-related columns
        health_cols = ['CD4 cell count (cells/µL)', 'Hemoglobin (g/dL)', 'Creatinine (mg/dL)']
        
        for col in health_cols:
            if col in self.data.columns:
                valid_data = self.data[col].notna()
                if valid_data.sum() > 100:  # Need at least 100 valid values
                    targets[col] = self.data[col]
                    print(f"✅ Target created: {col} ({valid_data.sum():,} valid values)")
        
        # Create composite health risk score if biomarkers available
        if len(targets) > 0:
            print("\n🔬 Creating composite health risk score...")
            risk_score = pd.Series(0, index=self.data.index)
            n_components = 0
            
            for col, values in targets.items():
                if values.notna().any():
                    # Normalize and add to risk score
                    normalized = (values - values.mean()) / values.std()
                    risk_score += normalized.fillna(0)
                    n_components += 1
            
            if n_components > 0:
                risk_score = risk_score / n_components
                targets['health_risk_score'] = risk_score
                print(f"✅ Composite health risk score created from {n_components} biomarkers")
        
        # Create climate vulnerability indicator if climate data available
        climate_cols = [col for col in self.preprocessed_data.columns if 
                       any(word in col.lower() for word in ['temp', 'heat', 'climate'])]
        
        if climate_cols:
            print("\n🌡️ Creating climate exposure indicator...")
            climate_exposure = self.preprocessed_data[climate_cols].mean(axis=1)
            targets['climate_exposure'] = climate_exposure
            print(f"✅ Climate exposure indicator created from {len(climate_cols)} variables")
        
        return targets
    
    def run_ml_analysis(self, target_name, target_values):
        """Run comprehensive ML analysis for a specific target."""
        print(f"\n🔬 Analyzing: {target_name}")
        print("-" * 40)
        
        # Prepare data
        valid_idx = target_values.notna()
        X = self.preprocessed_data.loc[valid_idx]
        y = target_values.loc[valid_idx]
        
        if len(X) < 100:
            print(f"   ⚠️  Insufficient data: {len(X)} samples")
            return None
        
        print(f"   📊 Samples: {len(X):,}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, 
                min_samples_split=20, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42
            ),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"   🤖 Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
            
            results[model_name] = {
                'model': model,
                'scaler': scaler,
                'r2': r2,
                'rmse': rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_names': X.columns.tolist()
            }
            
            print(f"      R²: {r2:.3f} (CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_result = results[best_model_name]
        
        print(f"   🏆 Best model: {best_model_name} (R² = {best_result['r2']:.3f})")
        
        return best_result
    
    def generate_shap_explanations(self, model_result, target_name, sample_size=100):
        """Generate SHAP explanations for model interpretability."""
        print(f"\n🔍 Generating SHAP explanations for {target_name}...")
        
        model = model_result['model']
        scaler = model_result['scaler']
        feature_names = model_result['feature_names']
        
        # Get a sample of data for SHAP
        valid_idx = self.data[target_name].notna() if target_name in self.data.columns else self.preprocessed_data.index
        X_sample = self.preprocessed_data.loc[valid_idx].sample(
            n=min(sample_size, len(valid_idx)), random_state=42
        )
        X_sample_scaled = scaler.transform(X_sample)
        
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, X_sample_scaled)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Store results
        self.shap_values[target_name] = {
            'values': shap_values,
            'features': feature_names,
            'data': X_sample_scaled
        }
        
        print(f"   ✅ SHAP analysis complete for {sample_size} samples")
        
        return shap_values
    
    def create_visualizations(self):
        """Create publication-quality visualizations."""
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create figures directory
        Path("figures/scaled_analysis").mkdir(parents=True, exist_ok=True)
        
        # 1. Data distribution overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Data source distribution
        if 'data_source' in self.data.columns:
            source_counts = self.data['data_source'].value_counts()
            axes[0, 0].pie(source_counts.values, labels=source_counts.index, 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Data Source Distribution')
        
        # Temporal distribution
        if 'survey_year' in self.data.columns:
            year_counts = self.data['survey_year'].value_counts().sort_index()
            axes[0, 1].bar(year_counts.index, year_counts.values, color='skyblue', edgecolor='navy')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Records')
            axes[0, 1].set_title('Temporal Distribution')
        
        # Feature completeness
        if self.preprocessed_data is not None and len(self.preprocessed_data) > 0:
            completeness = (self.preprocessed_data.notna().sum() / len(self.preprocessed_data) * 100).sort_values()
            top_features = completeness.tail(20)
            axes[1, 0].barh(range(len(top_features)), top_features.values)
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels([name[:30] for name in top_features.index], fontsize=8)
            axes[1, 0].set_xlabel('Completeness (%)')
            axes[1, 0].set_title('Top 20 Features by Completeness')
        
        # Model performance comparison
        if self.results:
            model_names = []
            r2_scores = []
            for target, result in self.results.items():
                if result:
                    model_names.append(target[:20])
                    r2_scores.append(result['r2'])
            
            if model_names:
                axes[1, 1].bar(model_names, r2_scores, color='green', alpha=0.7)
                axes[1, 1].set_xlabel('Target Variable')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].set_title('Model Performance')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('HEAT Center Scaled ML Analysis Overview', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save as SVG
        output_path = "figures/scaled_analysis/overview_dashboard.svg"
        plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
        
        # 2. SHAP summary plots
        for target_name, shap_data in self.shap_values.items():
            fig = plt.figure(figsize=(12, 8))
            
            # Create summary plot
            shap.summary_plot(
                shap_data['values'], 
                features=shap_data['data'],
                feature_names=shap_data['features'][:20],  # Top 20 features
                show=False
            )
            
            plt.title(f'SHAP Feature Importance: {target_name}', fontsize=14)
            
            # Save
            output_path = f"figures/scaled_analysis/shap_{target_name.replace('/', '_')}.svg"
            plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {output_path}")
            plt.close()
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("📋 GENERATING ANALYSIS REPORT")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': len(self.data) if self.data is not None else 0,
                'total_features': len(self.preprocessed_data.columns) if self.preprocessed_data is not None else 0,
                'data_sources': {}
            },
            'model_results': {},
            'key_findings': []
        }
        
        # Data source breakdown
        if self.data is not None and 'data_source' in self.data.columns:
            source_counts = self.data['data_source'].value_counts()
            for source, count in source_counts.items():
                report['dataset_info']['data_sources'][source] = int(count)
        
        # Model results
        for target_name, result in self.results.items():
            if result:
                report['model_results'][target_name] = {
                    'r2_score': float(result['r2']),
                    'rmse': float(result['rmse']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
        
        # Key findings
        if report['model_results']:
            best_target = max(report['model_results'].keys(), 
                            key=lambda k: report['model_results'][k]['r2_score'])
            best_r2 = report['model_results'][best_target]['r2_score']
            
            report['key_findings'].append(
                f"Best predictive model: {best_target} (R² = {best_r2:.3f})"
            )
            report['key_findings'].append(
                f"Successfully analyzed {len(report['model_results'])} target variables"
            )
            report['key_findings'].append(
                f"Integrated {sum(report['dataset_info']['data_sources'].values()):,} records from multiple sources"
            )
        
        # Save report
        report_path = Path("results/scaled_ml_analysis_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved: {report_path}")
        
        # Print summary
        print("\n📊 ANALYSIS SUMMARY:")
        print(f"   • Records analyzed: {report['dataset_info']['total_records']:,}")
        print(f"   • Features used: {report['dataset_info']['total_features']:,}")
        print(f"   • Models trained: {len(report['model_results'])}")
        
        if report['key_findings']:
            print("\n🔍 Key Findings:")
            for finding in report['key_findings']:
                print(f"   • {finding}")
        
        return report
    
    def run_complete_analysis(self, sample_fraction=0.5):
        """Run the complete scaled ML analysis pipeline."""
        print("=" * 60)
        print("🚀 SCALED ML ANALYSIS PIPELINE")
        print(f"Processing massive integrated dataset")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 1. Load data
        self.load_and_sample_data(sample_fraction=sample_fraction)
        
        # 2. Preprocess
        self.preprocess_data()
        
        if self.preprocessed_data is None or len(self.preprocessed_data) == 0:
            print("❌ No data available for analysis")
            return None
        
        # 3. Create targets
        targets = self.create_analysis_targets()
        
        # 4. Run ML analysis for each target
        for target_name, target_values in targets.items():
            if target_values is not None and target_values.notna().sum() > 100:
                result = self.run_ml_analysis(target_name, target_values)
                if result:
                    self.results[target_name] = result
                    
                    # Generate SHAP explanations for best models
                    if result['r2'] > 0.3:  # Only for reasonably good models
                        self.generate_shap_explanations(result, target_name)
        
        # 5. Create visualizations
        self.create_visualizations()
        
        # 6. Generate report
        self.generate_analysis_report()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"📊 Models trained: {len(self.results)}")
        print(f"📁 Results saved to: results/ and figures/scaled_analysis/")
        print("=" * 60)
        
        return self.results

def main():
    """Main function to run scaled ML analysis."""
    print("🌟 HEAT CENTER SCALED ML ANALYSIS")
    print("Analyzing 128k+ integrated records with advanced ML techniques")
    print()
    
    # Initialize pipeline
    pipeline = ScaledMLPipeline()
    
    # Run analysis (using 50% sample for efficiency)
    # Set to 1.0 to use full dataset
    results = pipeline.run_complete_analysis(sample_fraction=0.5)
    
    if results:
        print("\n✅ SUCCESS: Scaled ML analysis complete!")
        print(f"   📊 Analyzed {len(results)} target variables")
        print("   📁 Check results/ and figures/scaled_analysis/ for outputs")
        return True
    else:
        print("\n❌ Analysis failed - please check error messages")
        return False

if __name__ == "__main__":
    main()