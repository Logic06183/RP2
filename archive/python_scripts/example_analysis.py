#!/usr/bin/env python
"""
Example Heat-Health XAI Analysis

Demonstrates how to use the heat-health XAI framework for multi-dataset analysis.
"""

from pathlib import Path
from core.pipeline import HeatHealthAnalyzer

def main():
    """Run example analysis on DPHRU053 dataset."""
    
    print("🚀 Heat-Health XAI Framework Example")
    print("=" * 50)
    
    # Initialize analyzer with DPHRU053 dataset
    analyzer = HeatHealthAnalyzer(dataset_name='dphru053')
    
    # Run complete analysis
    results = analyzer.run_analysis(
        pathways=['inflammatory', 'metabolic', 'cardiovascular'],
        use_temporal_features=True,
        use_interaction_features=True,
        explain_predictions=True
    )
    
    # Generate comprehensive report
    output_dir = analyzer.generate_report(results)
    
    print(f"✅ Analysis complete! Results saved to: {output_dir}")
    
    # Print summary of findings
    print("\n📊 Summary of Findings:")
    print("-" * 30)
    
    for pathway, pathway_results in results['pathways'].items():
        metrics = pathway_results['metrics']
        print(f"\n{pathway.title()} Pathway:")
        print(f"  R² Score: {metrics['test_r2']:.3f}")
        print(f"  RMSE: {metrics['test_rmse']:.3f}")
        print(f"  Features: {metrics['n_features']}")
        print(f"  Samples: {metrics['n_samples']}")
        
        if 'xai_results' in pathway_results and pathway_results['xai_results']:
            print(f"  XAI Analysis: ✅ Available")
            if 'hypotheses' in pathway_results['xai_results']:
                hypotheses = pathway_results['xai_results']['hypotheses']
                print(f"  Novel Hypotheses: {len(hypotheses)} generated")
        else:
            print(f"  XAI Analysis: ❌ Low predictive performance")

def multi_dataset_example():
    """Example of comparing across multiple datasets."""
    
    print("\n🔄 Multi-Dataset Comparison Example")
    print("=" * 40)
    
    datasets = ['dphru053']  # Add more dataset names as they become available
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n📊 Analyzing {dataset}...")
        
        analyzer = HeatHealthAnalyzer(dataset_name=dataset)
        results = analyzer.run_analysis(
            pathways=['inflammatory', 'metabolic'],
            explain_predictions=False  # Skip XAI for quick comparison
        )
        
        all_results[dataset] = results
    
    # Compare results across datasets
    print("\n🔍 Cross-Dataset Comparison:")
    print("-" * 30)
    
    for pathway in ['inflammatory', 'metabolic']:
        print(f"\n{pathway.title()} Pathway Performance:")
        for dataset, results in all_results.items():
            if pathway in results['pathways']:
                r2 = results['pathways'][pathway]['metrics']['test_r2']
                print(f"  {dataset}: R² = {r2:.3f}")

def custom_config_example():
    """Example of using custom configuration."""
    
    print("\n⚙️  Custom Configuration Example")
    print("=" * 35)
    
    # Custom analysis parameters
    custom_config = {
        'lag_periods': [7, 14, 21],  # Shorter lag periods
        'cv_folds': 3,               # Fewer CV folds for speed
        'test_size': 0.2,            # Smaller test set
        'shap_sample_size': 500      # Smaller SHAP sample
    }
    
    analyzer = HeatHealthAnalyzer(
        dataset_name='dphru053',
        custom_config=custom_config
    )
    
    results = analyzer.run_analysis(
        pathways=['inflammatory'],
        use_temporal_features=True,
        explain_predictions=True
    )
    
    print("✅ Custom configuration analysis complete!")

if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to run additional examples
    # multi_dataset_example()
    # custom_config_example()