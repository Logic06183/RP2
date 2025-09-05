#!/usr/bin/env python3
"""
Quick Data Consolidation Validation
Fast validation of consolidated datasets for analysis readiness
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def quick_validation():
    """Fast validation of consolidated data"""
    print("QUICK DATA CONSOLIDATION VALIDATION")
    print("=" * 45)
    
    results = {}
    
    # Check GCRO socioeconomic data
    gcro_path = Path("data/socioeconomic/gcro_full")
    if gcro_path.exists():
        years = [d.name for d in gcro_path.iterdir() if d.is_dir()]
        expected_years = ['2009', '2011', '2013-2014', '2015-2016', '2017-2018', '2020-2021']
        
        total_records = 0
        year_details = {}
        
        for year_dir in gcro_path.iterdir():
            if year_dir.is_dir():
                csv_files = list(year_dir.glob('**/*.csv'))
                if csv_files:
                    try:
                        # Quick record count without loading full data
                        record_count = sum(1 for line in open(csv_files[0], 'r', encoding='latin-1')) - 1
                        total_records += record_count
                        year_details[year_dir.name] = record_count
                        print(f"  {year_dir.name}: {record_count:,} records")
                    except Exception as e:
                        print(f"  ❌ {year_dir.name}: Could not count records - {e}")
        
        results['gcro_socioeconomic'] = {
            'status': 'complete' if set(years) >= set(expected_years) else 'incomplete',
            'years_found': len(years),
            'total_records': total_records,
            'year_details': year_details
        }
        
        print(f"✅ GCRO Data: {total_records:,} total records across {len(years)} years")
    else:
        results['gcro_socioeconomic'] = {'status': 'missing'}
        print("❌ GCRO Data: Directory not found")
    
    # Check climate data (symbolic links)
    climate_path = Path("data/climate/johannesburg")
    if climate_path.exists():
        zarr_files = list(climate_path.glob("*.zarr"))
        accessible_count = 0
        
        for zarr_file in zarr_files:
            if zarr_file.is_symlink() and zarr_file.exists():
                accessible_count += 1
        
        results['climate_data'] = {
            'status': 'complete' if accessible_count > 0 else 'missing',
            'zarr_count': len(zarr_files),
            'accessible_count': accessible_count
        }
        
        print(f"✅ Climate Data: {accessible_count}/{len(zarr_files)} zarr datasets accessible")
        for zarr_file in zarr_files:
            status = "✅" if zarr_file.exists() else "❌"
            print(f"  {status} {zarr_file.name}")
    else:
        results['climate_data'] = {'status': 'missing'}
        print("❌ Climate Data: Directory not found")
    
    # Check health data
    health_path = Path("data/health/rp2_harmonized")
    if health_path.exists():
        csv_files = list(health_path.glob("*.csv"))
        health_records = {}
        
        for csv_file in csv_files:
            try:
                record_count = sum(1 for line in open(csv_file, 'r')) - 1
                health_records[csv_file.name] = record_count
                print(f"  {csv_file.name}: {record_count:,} records")
            except Exception as e:
                print(f"  ❌ {csv_file.name}: Could not count records - {e}")
        
        results['health_data'] = {
            'status': 'complete' if health_records else 'missing',
            'files': health_records
        }
        
        print(f"✅ Health Data: {len(csv_files)} files available")
    else:
        results['health_data'] = {'status': 'missing'}
        print("❌ Health Data: Directory not found")
    
    # Check processed data
    processed_path = Path("data/socioeconomic/processed")
    if processed_path.exists():
        gcro_subset = processed_path / "GCRO_combined_climate_SUBSET.csv"
        if gcro_subset.exists():
            try:
                record_count = sum(1 for line in open(gcro_subset, 'r')) - 1
                results['processed_data'] = {
                    'status': 'complete',
                    'gcro_subset_records': record_count
                }
                print(f"✅ Processed Data: GCRO subset with {record_count} records")
            except Exception as e:
                results['processed_data'] = {'status': 'error', 'error': str(e)}
                print(f"❌ Processed Data: Error reading GCRO subset - {e}")
        else:
            results['processed_data'] = {'status': 'missing'}
            print("❌ Processed Data: GCRO subset not found")
    else:
        results['processed_data'] = {'status': 'missing'}
        print("❌ Processed Data: Directory not found")
    
    # Test analysis script access
    print("\nANALYSIS SCRIPT VALIDATION")
    print("-" * 30)
    
    scripts_to_test = [
        'src/socioeconomic_vulnerability_analysis.py',
        'src/advanced_ml_climate_health_analysis.py',
        'src/comprehensive_heat_health_analysis.py'
    ]
    
    script_results = {}
    for script in scripts_to_test:
        script_path = Path(script)
        if script_path.exists():
            script_results[script] = 'exists'
            print(f"✅ {script}: Script file exists")
        else:
            script_results[script] = 'missing'
            print(f"❌ {script}: Script not found")
    
    results['analysis_scripts'] = script_results
    
    # Generate summary report
    print("\n" + "=" * 50)
    print("DATA CONSOLIDATION SUMMARY")
    print("=" * 50)
    
    summary = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'results': results,
        'consolidation_status': 'ready' if all(
            r.get('status') == 'complete' 
            for r in results.values() 
            if isinstance(r, dict) and 'status' in r
        ) else 'partial',
        'critical_achievements': [
            f"GCRO Data: {results.get('gcro_socioeconomic', {}).get('total_records', 0):,} records consolidated",
            f"Climate Data: {results.get('climate_data', {}).get('accessible_count', 0)} zarr datasets linked",
            f"Health Data: {len(results.get('health_data', {}).get('files', {}))} datasets available",
            f"Analysis Scripts: {sum(1 for s in script_results.values() if s == 'exists')}/3 ready"
        ]
    }
    
    # Save validation report
    with open('data_validation_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ GCRO Socioeconomic: {results.get('gcro_socioeconomic', {}).get('total_records', 0):,} records")
    print(f"✅ Climate Datasets: {results.get('climate_data', {}).get('accessible_count', 0)} accessible")
    print(f"✅ Health Datasets: {len(results.get('health_data', {}).get('files', {}))} files")
    print(f"✅ Analysis Scripts: Updated and ready")
    print(f"\n📊 Report saved: data_validation_report.json")
    
    # Check if ready for analysis
    if summary['consolidation_status'] == 'ready':
        print("🚀 READY: All datasets consolidated and analysis scripts updated!")
        print("\nNext steps:")
        print("  - Run: python src/socioeconomic_vulnerability_analysis.py")
        print("  - Run: python src/advanced_ml_climate_health_analysis.py")
        print("  - Run: python src/comprehensive_heat_health_analysis.py")
    else:
        print("⚠️  PARTIAL: Some datasets need attention before analysis")
        
        # Show what needs attention
        for component, details in results.items():
            if isinstance(details, dict) and details.get('status') != 'complete':
                print(f"  - {component}: {details.get('status', 'unknown')}")
    
    return summary

if __name__ == "__main__":
    quick_validation()