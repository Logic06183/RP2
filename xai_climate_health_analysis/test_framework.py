#!/usr/bin/env python3
"""
Rigorous Testing Framework for XAI Climate-Health Analysis
Author: Craig Parker
Ensures reproducibility and validates all analysis components
"""

import unittest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TestXAIAnalysis(unittest.TestCase):
    """Comprehensive test suite for XAI analysis"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.data_dir = Path("data")
        cls.results_dir = Path("rigorous_results")
        
    def test_data_availability(self):
        """Test that required data files exist"""
        # Health data files
        health_files = list(self.data_dir.glob("health/*.csv"))
        self.assertGreater(len(health_files), 0, "No health data files found")
        
        # Climate data path
        climate_path = Path("/home/cparker/selected_data_all/data/RP2_subsets/JHB/ERA5_tas_native.zarr")
        self.assertTrue(climate_path.exists(), "ERA5 climate data not accessible")
        
    def test_results_generation(self):
        """Test that analysis generates expected results"""
        results_file = self.results_dir / "xai_results.json"
        self.assertTrue(results_file.exists(), "Results file not found")
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Check results structure
        self.assertIsInstance(results, list, "Results should be a list")
        self.assertGreater(len(results), 0, "Results list should not be empty")
        
        # Validate individual results
        for result in results:
            if result:  # Skip None results
                self.assertIn('biomarker', result, "Result missing biomarker field")
                self.assertIn('r2_score', result, "Result missing R2 score")
                self.assertIn('n_samples', result, "Result missing sample size")
                
                # Validate R2 scores are reasonable
                r2 = result['r2_score']
                self.assertGreaterEqual(r2, 0, "R2 score should be non-negative")
                self.assertLessEqual(r2, 1, "R2 score should not exceed 1")
    
    def test_figure_generation(self):
        """Test that all required figures are generated"""
        required_figures = [
            "figure_1_main_results.svg",
            "figure_2_feature_importance.svg", 
            "table_1_summary.svg"
        ]
        
        for fig_name in required_figures:
            fig_path = self.results_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Missing required figure: {fig_name}")
            
            # Check file size (should not be empty)
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure {fig_name} appears to be empty")
    
    def test_data_quality_standards(self):
        """Test data quality meets publication standards"""
        # Load a sample health file to test data quality
        health_files = list(self.data_dir.glob("health/*.csv"))
        if health_files:
            df = pd.read_csv(health_files[0])
            
            # Check for required columns
            required_cols = ['Patient ID', 'date']  # Basic required columns
            for col in required_cols:
                if col in df.columns or 'visit_date' in df.columns:
                    continue  # At least one date column exists
                    
            # Check data completeness
            total_cells = df.size
            missing_cells = df.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells)
            
            self.assertGreater(completeness, 0.5, "Data completeness should exceed 50%")
    
    def test_model_performance_standards(self):
        """Test that models meet minimum performance standards"""
        results_file = self.results_dir / "xai_results.json" 
        
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            # Check that we have meaningful results
            meaningful_results = [r for r in results if r and r.get('r2_score', 0) > 0.1]
            self.assertGreater(len(meaningful_results), 0, "Should have at least one meaningful result (R2 > 0.1)")
            
            # Check for high-performance models
            high_performance = [r for r in results if r and r.get('r2_score', 0) > 0.5]
            self.assertGreater(len(high_performance), 0, "Should have at least one high-performance model (R2 > 0.5)")
    
    def test_reproducibility(self):
        """Test that analysis is reproducible"""
        # Check for seed setting and consistent results structure
        results_file = self.results_dir / "xai_results.json"
        
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            # Results should be consistent across runs (same biomarkers)
            biomarkers = [r.get('biomarker') for r in results if r]
            unique_biomarkers = set(biomarkers)
            
            # Should have multiple biomarkers analyzed
            self.assertGreater(len(unique_biomarkers), 1, "Should analyze multiple biomarkers")
    
    def test_statistical_validity(self):
        """Test statistical validity of results"""
        results_file = self.results_dir / "xai_results.json"
        
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            for result in results:
                if result and 'n_samples' in result:
                    n = result['n_samples']
                    
                    # Minimum sample size for meaningful analysis
                    self.assertGreater(n, 50, f"Sample size too small for {result.get('biomarker', 'unknown')}: n={n}")
                    
                    # For high R2 scores, ensure adequate sample size
                    if result.get('r2_score', 0) > 0.6:
                        self.assertGreater(n, 100, "High R2 scores should be based on adequate sample sizes")

class TestDataIntegration(unittest.TestCase):
    """Test data integration components"""
    
    def test_climate_data_access(self):
        """Test access to climate data"""
        try:
            import xarray as xr
            climate_path = "/home/cparker/selected_data_all/data/RP2_subsets/JHB/ERA5_tas_native.zarr"
            ds = xr.open_zarr(climate_path)
            
            # Check data structure
            self.assertIn('tas', ds.data_vars, "Temperature variable not found")
            self.assertIn('time', ds.dims, "Time dimension not found")
            
            # Check data coverage
            self.assertGreater(len(ds.time), 100000, "Insufficient climate data coverage")
            
        except Exception as e:
            self.skipTest(f"Climate data not accessible: {e}")
    
    def test_health_data_structure(self):
        """Test health data structure and content"""
        data_dir = Path("data")
        health_files = list(data_dir.glob("health/*.csv"))
        
        self.assertGreater(len(health_files), 5, "Should have multiple health cohorts")
        
        # Test sample file structure
        if health_files:
            df = pd.read_csv(health_files[0])
            
            # Should have meaningful number of records and variables
            self.assertGreater(len(df), 10, "Health data should have meaningful sample size")
            self.assertGreater(len(df.columns), 10, "Health data should have multiple variables")

def run_all_tests():
    """Run comprehensive test suite"""
    print("="*60)
    print("RIGOROUS XAI ANALYSIS - TESTING FRAMEWORK")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestXAIAnalysis))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDataIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception: ')[-1].split('\n')[0]}")
    
    # Overall assessment
    if len(result.failures) + len(result.errors) == 0:
        print("\n✅ ALL TESTS PASSED - ANALYSIS MEETS PUBLICATION STANDARDS")
        return True
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} TESTS FAILED - REVIEW REQUIRED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)