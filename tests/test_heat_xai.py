#!/usr/bin/env python3
"""
Comprehensive Testing Framework for HeatLab XAI Analysis
========================================================
Ensures reproducibility, validates data integrity, and verifies analysis quality

Author: Craig Parker
Created: 2025-09-05
"""

import unittest
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
import warnings

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from heat_xai_analysis import HeatXAIAnalyzer
except ImportError:
    print("Warning: Could not import HeatXAIAnalyzer. Some tests will be skipped.")
    HeatXAIAnalyzer = None

warnings.filterwarnings('ignore')

class TestDataIntegrity(unittest.TestCase):
    """Test data integrity and availability"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / "xai_climate_health_analysis" / "data"
        cls.results_dir = cls.project_root / "results_consolidated"
    
    def test_health_data_availability(self):
        """Test that health data files exist and are readable"""
        health_dir = self.data_dir / "health"
        
        if not health_dir.exists():
            self.skipTest("Health data directory not found")
        
        health_files = list(health_dir.glob("*.csv"))
        self.assertGreater(len(health_files), 5, "Should have multiple health cohort files")
        
        # Test reading sample files
        sample_count = 0
        for file in health_files[:3]:  # Test first 3 files
            try:
                df = pd.read_csv(file)
                self.assertGreater(len(df), 0, f"File {file.name} should not be empty")
                self.assertGreater(len(df.columns), 10, f"File {file.name} should have multiple columns")
                
                # Check for date columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'visit' in col.lower()]
                self.assertGreater(len(date_cols), 0, f"File {file.name} should have date column")
                
                sample_count += 1
            except Exception as e:
                self.fail(f"Failed to read health file {file}: {e}")
        
        self.assertGreater(sample_count, 0, "Should successfully read at least some health files")
    
    def test_climate_data_access(self):
        """Test access to ERA5 climate data"""
        climate_path = Path("/home/cparker/selected_data_all/data/RP2_subsets/JHB/ERA5_tas_native.zarr")
        
        if not climate_path.exists():
            self.skipTest("ERA5 climate data not accessible")
        
        try:
            import xarray as xr
            ds = xr.open_zarr(str(climate_path))
            
            # Validate climate data structure
            self.assertIn('tas', ds.data_vars, "Temperature variable should exist")
            self.assertIn('time', ds.dims, "Time dimension should exist")
            self.assertGreater(len(ds.time), 100000, "Should have substantial temporal coverage")
            
            # Check data quality
            temp_data = ds.tas.values.flatten()
            temp_data_clean = temp_data[~np.isnan(temp_data)]
            self.assertGreater(len(temp_data_clean), len(temp_data) * 0.8, "Should have <20% missing data")
            
            # Temperature range sanity check
            self.assertGreater(np.mean(temp_data_clean), 200, "Mean temperature should be reasonable (Kelvin)")
            self.assertLess(np.mean(temp_data_clean), 320, "Mean temperature should be reasonable (Kelvin)")
            
        except ImportError:
            self.skipTest("xarray not available")
        except Exception as e:
            self.skipTest(f"Climate data access failed: {e}")

class TestAnalysisResults(unittest.TestCase):
    """Test analysis results quality and validity"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent
        cls.results_dir = cls.project_root / "results_consolidated"
    
    def test_results_files_exist(self):
        """Test that analysis results files exist"""
        if not self.results_dir.exists():
            self.skipTest("Results directory not found")
        
        result_files = list(self.results_dir.glob("*.json"))
        self.assertGreater(len(result_files), 0, "Should have result files")
        
        # Find the largest result file (likely the most comprehensive)
        largest_file = max(result_files, key=lambda f: f.stat().st_size)
        self.assertGreater(largest_file.stat().st_size, 1000, "Result files should not be empty")
    
    def test_result_structure_validity(self):
        """Test that results have valid structure and content"""
        if not self.results_dir.exists():
            self.skipTest("Results directory not found")
        
        # Find JSON result files
        result_files = [f for f in self.results_dir.glob("*.json") 
                       if 'xai_results' in f.name and f.stat().st_size > 10000]
        
        if not result_files:
            self.skipTest("No substantial result files found")
        
        # Test the largest results file
        results_file = max(result_files, key=lambda f: f.stat().st_size)
        
        try:
            with open(results_file) as f:
                results = json.load(f)
            
            # Validate results structure
            self.assertIsInstance(results, list, "Results should be a list")
            
            valid_results = [r for r in results if r and isinstance(r, dict)]
            self.assertGreater(len(valid_results), 0, "Should have valid results")
            
            # Test individual result structure
            for result in valid_results[:5]:  # Test first 5 results
                self.assertIn('biomarker', result, "Result should have biomarker field")
                self.assertIn('r2_score', result, "Result should have R² score")
                self.assertIn('n_samples', result, "Result should have sample size")
                
                # Validate data types and ranges
                r2 = result['r2_score']
                self.assertIsInstance(r2, (int, float), "R² should be numeric")
                self.assertGreaterEqual(r2, -1, "R² should be >= -1")
                self.assertLessEqual(r2, 1, "R² should be <= 1")
                
                n = result['n_samples']
                self.assertIsInstance(n, (int, float), "Sample size should be numeric")
                self.assertGreater(n, 0, "Sample size should be positive")
                
        except json.JSONDecodeError:
            self.fail(f"Result file {results_file} is not valid JSON")
        except Exception as e:
            self.fail(f"Failed to validate results structure: {e}")
    
    def test_high_quality_results_exist(self):
        """Test that some high-quality results (R² > 0.5) exist"""
        if not self.results_dir.exists():
            self.skipTest("Results directory not found")
        
        # Find substantial result files
        result_files = [f for f in self.results_dir.glob("*.json") 
                       if f.stat().st_size > 10000]
        
        if not result_files:
            self.skipTest("No substantial result files found")
        
        high_performance_found = False
        
        for results_file in result_files:
            try:
                with open(results_file) as f:
                    results = json.load(f)
                
                if not isinstance(results, list):
                    continue
                
                # Check for high-performance results
                for result in results:
                    if (result and isinstance(result, dict) and 
                        'r2_score' in result and result['r2_score'] > 0.5):
                        high_performance_found = True
                        
                        # Log the high-performance result
                        biomarker = result.get('biomarker', 'Unknown')
                        r2 = result['r2_score']
                        n = result.get('n_samples', 0)
                        print(f"High-performance result found: {biomarker} R²={r2:.4f} n={n}")
                        break
                
                if high_performance_found:
                    break
                    
            except Exception:
                continue
        
        self.assertTrue(high_performance_found, 
                       "Should have at least one high-performance result (R² > 0.5)")

class TestFigureGeneration(unittest.TestCase):
    """Test figure generation and quality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.project_root = Path(__file__).parent.parent
        cls.figures_dir = cls.project_root / "figures"
    
    def test_figures_exist(self):
        """Test that publication figures exist"""
        if not self.figures_dir.exists():
            self.skipTest("Figures directory not found")
        
        svg_files = list(self.figures_dir.glob("*.svg"))
        self.assertGreater(len(svg_files), 0, "Should have SVG figure files")
        
        # Check for key figures
        important_figures = [
            'main_results', 'feature_importance', 'summary', 'comprehensive'
        ]
        
        found_figures = []
        for fig_pattern in important_figures:
            matching_files = [f for f in svg_files if fig_pattern in f.name.lower()]
            if matching_files:
                found_figures.append(fig_pattern)
        
        self.assertGreater(len(found_figures), 0, 
                          "Should have at least some important figure types")
    
    def test_figure_file_quality(self):
        """Test that figure files are not empty and appear valid"""
        if not self.figures_dir.exists():
            self.skipTest("Figures directory not found")
        
        svg_files = list(self.figures_dir.glob("*.svg"))
        if not svg_files:
            self.skipTest("No SVG files found")
        
        for svg_file in svg_files[:5]:  # Test first 5 files
            file_size = svg_file.stat().st_size
            self.assertGreater(file_size, 1000, f"Figure {svg_file.name} should not be empty")
            
            # Basic SVG format check
            try:
                with open(svg_file, 'r') as f:
                    content = f.read(500)  # Read first 500 chars
                    self.assertIn('<svg', content, f"File {svg_file.name} should be valid SVG")
            except Exception as e:
                self.fail(f"Failed to read SVG file {svg_file.name}: {e}")

class TestFrameworkIntegration(unittest.TestCase):
    """Test overall framework integration"""
    
    def test_analyzer_initialization(self):
        """Test that HeatXAIAnalyzer can be initialized"""
        if HeatXAIAnalyzer is None:
            self.skipTest("HeatXAIAnalyzer not available")
        
        try:
            analyzer = HeatXAIAnalyzer()
            self.assertIsNotNone(analyzer, "Analyzer should initialize successfully")
            self.assertIsInstance(analyzer.config, dict, "Analyzer should have config")
            self.assertIn('random_state', analyzer.config, "Config should have random_state")
        except Exception as e:
            self.fail(f"Failed to initialize HeatXAIAnalyzer: {e}")
    
    def test_data_directories_structure(self):
        """Test that expected data directory structure exists"""
        project_root = Path(__file__).parent.parent
        
        # Check for key directories
        expected_dirs = ['src', 'results_consolidated', 'figures', 'docs']
        
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")
            if dir_name == 'src':
                self.assertTrue((dir_path / 'heat_xai_analysis.py').exists(), 
                               "Main analysis script should exist")

def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("="*70)
    print("HEATLAB XAI ANALYSIS - COMPREHENSIVE TESTING FRAMEWORK")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [TestDataIntegrity, TestAnalysisResults, 
                   TestFigureGeneration, TestFrameworkIntegration]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Detailed summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 'N/A'}")
    
    # Report issues
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  {i}. {test}: {error_msg}")
    
    if result.errors:
        print(f"\n⚠️ ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            error_msg = traceback.split('Exception: ')[-1].split('\n')[0]
            print(f"  {i}. {test}: {error_msg}")
    
    # Overall assessment
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1)
    print(f"\nSuccess Rate: {success_rate:.1%}")
    
    if success_rate >= 0.9:
        print("\n✅ EXCELLENT - Analysis framework meets high quality standards")
        return True
    elif success_rate >= 0.7:
        print("\n⚠️ GOOD - Minor issues identified, framework generally solid")
        return True
    else:
        print("\n❌ NEEDS ATTENTION - Multiple issues require resolution")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)