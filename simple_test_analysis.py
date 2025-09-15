#!/usr/bin/env python3
"""
Simple test analysis script for demonstrating the robust analysis system.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    """Run a simple test analysis."""
    print("🔥 Starting Simple Test Analysis")
    
    # Create some synthetic data
    np.random.seed(42)
    
    # Generate synthetic temperature and health data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    temperatures = 25 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + np.random.normal(0, 3, 365)
    health_outcomes = np.random.poisson(10 + 0.1 * (temperatures - 25), 365)
    
    data = pd.DataFrame({
        'date': dates,
        'temperature': temperatures,
        'health_outcomes': health_outcomes
    })
    
    print(f"📊 Generated {len(data)} days of synthetic data")
    
    # Basic analysis
    correlation = data['temperature'].corr(data['health_outcomes'])
    mean_temp = data['temperature'].mean()
    mean_outcomes = data['health_outcomes'].mean()
    
    print(f"🌡️  Mean temperature: {mean_temp:.2f}°C")
    print(f"🏥 Mean health outcomes: {mean_outcomes:.2f}")
    print(f"🔗 Correlation: {correlation:.3f}")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Save analysis results
    results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'simple_test_analysis',
        'summary_statistics': {
            'mean_temperature': float(mean_temp),
            'mean_health_outcomes': float(mean_outcomes),
            'correlation': float(correlation),
            'data_points': len(data)
        },
        'data_period': {
            'start_date': str(data['date'].min().date()),
            'end_date': str(data['date'].max().date())
        }
    }
    
    # Save results
    with open('results/simple_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create a simple plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Temperature time series
    ax1.plot(data['date'], data['temperature'], 'b-', alpha=0.7)
    ax1.set_title('Temperature Time Series')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, alpha=0.3)
    
    # Health outcomes vs temperature scatter
    ax2.scatter(data['temperature'], data['health_outcomes'], alpha=0.5, s=20)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Health Outcomes')
    ax2.set_title(f'Health Outcomes vs Temperature (r = {correlation:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(data['temperature'], data['health_outcomes'], 1)
    p = np.poly1d(z)
    ax2.plot(data['temperature'], p(data['temperature']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('figures/simple_analysis_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary CSV
    summary_data = data.groupby(data['date'].dt.month).agg({
        'temperature': ['mean', 'std'],
        'health_outcomes': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    summary_data.columns = ['_'.join(col).strip() for col in summary_data.columns]
    summary_data.to_csv('results/monthly_summary.csv')
    
    print("✅ Analysis completed successfully")
    print("📁 Results saved to:")
    print("   - results/simple_analysis_results.json")
    print("   - results/monthly_summary.csv") 
    print("   - figures/simple_analysis_plot.png")
    
    # Simulate some processing time
    time.sleep(2)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)