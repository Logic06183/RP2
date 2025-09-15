#!/usr/bin/env python3
"""
HEAT Analysis Pipeline - Complete Pipeline Runner
================================================

This script runs the complete modular HEAT analysis pipeline:
1. Data Loading & Validation
2. Data Preprocessing & Feature Engineering
3. Machine Learning & XAI Analysis

Author: Craig Parker
Date: 2025-09-15
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Add pipeline modules to path
sys.path.append(str(Path(__file__).parent))

# Import pipeline modules
from data_loader import HeatDataLoader
from data_preprocessor import HeatDataPreprocessor
from ml_analyzer import HeatMLAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_execution.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Execute the complete HEAT analysis pipeline.
    """
    start_time = datetime.now()

    print("\n" + "="*80)
    print("🏥 HEAT CLIMATE-HEALTH ANALYSIS PIPELINE")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    try:
        # ================================================================
        # STEP 1: DATA LOADING & VALIDATION
        # ================================================================
        print("\n🔍 STEP 1: DATA LOADING & VALIDATION")
        print("-" * 50)

        loader = HeatDataLoader(data_dir="../data")

        # Load full dataset (CRITICAL: Never sample for publication analysis!)
        logger.info("Loading master integrated dataset...")
        df_raw = loader.load_master_dataset(sample_fraction=1.0)

        # Validate data structure
        logger.info("Validating data structure...")
        validation_report = loader.validate_data_structure()

        # Print validation summary
        loader.print_validation_summary()

        # Get cleaned dataset
        logger.info("Applying basic data cleaning...")
        df_clean = loader.get_clean_dataset()

        print(f"✅ Step 1 Complete: {len(df_clean):,} records loaded and validated")

        # ================================================================
        # STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
        # ================================================================
        print("\n🔧 STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
        print("-" * 50)

        preprocessor = HeatDataPreprocessor()

        # Run complete preprocessing pipeline
        logger.info("Starting preprocessing pipeline...")
        df_processed, preprocessing_report = preprocessor.preprocess_complete_pipeline(df_clean)

        # Extract target variables and features
        target_variables = preprocessing_report['target_variables']
        feature_cols = [col for col in df_processed.columns if col not in target_variables]

        print(f"✅ Step 2 Complete: {len(feature_cols)} features, {len(target_variables)} targets")
        print(f"   📊 Features: {preprocessing_report['features_engineered']} engineered")
        print(f"   🎯 Targets: {target_variables}")

        # ================================================================
        # STEP 3: MACHINE LEARNING & XAI ANALYSIS
        # ================================================================
        print("\n🤖 STEP 3: MACHINE LEARNING & XAI ANALYSIS")
        print("-" * 50)

        # Initialize ML analyzer
        analyzer = HeatMLAnalyzer(results_dir="../results")

        # Run comprehensive analysis
        logger.info("Starting comprehensive biomarker analysis...")
        analysis_results = analyzer.analyze_all_biomarkers(
            df_processed, target_variables, feature_cols
        )

        # Create visualizations
        logger.info("Creating summary visualizations...")
        analyzer.create_summary_visualizations(analysis_results)

        print(f"✅ Step 3 Complete: {len(analysis_results['model_results'])} models trained")

        # ================================================================
        # SUMMARY & RESULTS
        # ================================================================
        print("\n📊 PIPELINE EXECUTION SUMMARY")
        print("-" * 50)

        end_time = datetime.now()
        execution_time = end_time - start_time

        print(f"⏱️  Total Execution Time: {execution_time}")
        print(f"📈 Dataset Scale: {len(df_processed):,} records")
        print(f"🧬 Biomarkers Analyzed: {len(analysis_results['model_results'])}")

        # Performance summary
        if analysis_results['summary_statistics']:
            stats = analysis_results['summary_statistics']
            print(f"🏆 Best Performing Biomarker: {stats.get('best_performing_biomarker', 'N/A')}")
            print(f"📊 Average R²: {stats.get('average_r2', 0):.4f}")
            print(f"✅ Models Above R² > 0.5: {stats.get('models_above_threshold', 0)}")

        # Show top performing models
        print(f"\n🔝 TOP PERFORMING MODELS:")
        model_results = analysis_results['model_results']
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1]['final_r2'],
            reverse=True
        )

        for i, (biomarker, results) in enumerate(sorted_models[:5]):
            biomarker_short = biomarker.split('(')[0].strip()
            r2 = results['final_r2']
            n_samples = results['n_samples']
            performance = analyzer._classify_performance(r2)
            print(f"   {i+1}. {biomarker_short}: R² = {r2:.4f} (n = {n_samples:,}) - {performance}")

        print(f"\n💾 Results Saved To: ../results/")
        print(f"   📋 analysis_results.json")
        print(f"   📊 model_performance_summary.csv")
        print(f"   📈 model_performance_summary.png")

        print("\n" + "="*80)
        print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)

        return {
            'df_processed': df_processed,
            'analysis_results': analysis_results,
            'validation_report': validation_report,
            'preprocessing_report': preprocessing_report,
            'execution_time': execution_time
        }

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ PIPELINE EXECUTION FAILED")
        print(f"Error: {e}")
        print("Check pipeline_execution.log for details")
        raise


def run_quick_validation():
    """
    Run a quick validation of the pipeline with a small sample.
    For testing purposes only - NEVER use for actual analysis!
    """
    print("\n⚠️  QUICK VALIDATION MODE (TESTING ONLY)")
    print("🚨 WARNING: This is for pipeline testing only!")
    print("🚨 For actual analysis, use run_complete_pipeline()")

    # Test with small sample
    loader = HeatDataLoader(data_dir="../data")
    df_test = loader.load_master_dataset(sample_fraction=0.01)  # 1% sample for testing

    preprocessor = HeatDataPreprocessor()
    df_processed, _ = preprocessor.preprocess_complete_pipeline(df_test)

    print(f"✅ Pipeline validation successful with {len(df_processed):,} test records")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='HEAT Analysis Pipeline')
    parser.add_argument('--mode', choices=['full', 'validate'],
                       default='full',
                       help='Pipeline mode: full analysis or quick validation')

    args = parser.parse_args()

    if args.mode == 'validate':
        run_quick_validation()
    else:
        results = main()