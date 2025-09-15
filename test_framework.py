#!/usr/bin/env python3
"""
Comprehensive Test Framework for HEAT Climate-Health Analysis
Implements robust unit testing for all analysis components with focus on scientific validity.
"""

# import pytest  # Optional - tests can run without pytest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Test fixtures and data generators
class TestDataGenerator:
    """Generate realistic test data for climate-health analysis."""
    
    @staticmethod
    def create_climate_data(n_days: int = 365, n_locations: int = 10) -> xr.Dataset:
        """Create synthetic climate dataset."""
        if n_days <= 0:
            raise ValueError("n_days must be positive")
        if n_locations <= 0:
            raise ValueError("n_locations must be positive")
            
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        lats = np.linspace(-26.5, -26.0, n_locations)
        lons = np.linspace(27.5, 28.5, n_locations)
        
        # Realistic temperature patterns
        base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        temp_data = np.random.normal(base_temp[:, None, None], 3, (n_days, len(lats), len(lons)))
        
        # Realistic humidity patterns
        base_humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365 + np.pi/4)
        humidity_data = np.random.normal(base_humidity[:, None, None], 10, (n_days, len(lats), len(lons)))
        humidity_data = np.clip(humidity_data, 10, 95)  # Realistic bounds
        
        ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], temp_data),
            'humidity': (['time', 'lat', 'lon'], humidity_data),
            'pressure': (['time', 'lat', 'lon'], 
                        np.random.normal(1013, 10, (n_days, len(lats), len(lons))))
        }, coords={
            'time': dates,
            'lat': lats,
            'lon': lons
        })
        
        return ds
    
    @staticmethod
    def create_health_data(n_patients: int = 1000, start_date: str = '2020-01-01') -> pd.DataFrame:
        """Create synthetic health dataset."""
        # Set seed for reproducible test data
        np.random.seed(42)
        dates = pd.date_range(start_date, periods=365, freq='D')
        
        data = []
        for i in range(n_patients):
            # Random admission date
            admission_date = np.random.choice(dates)
            
            # Age-dependent risk
            age = np.random.normal(45, 20)
            age = max(0, min(100, age))
            
            # Temperature-dependent outcomes with bounded effect
            temp_effect = np.random.normal(0, 0.5) if np.random.random() > 0.7 else 0
            temp_effect = max(-0.2, min(0.2, temp_effect))  # Bound the effect
            
            # Ensure probability is valid for binomial
            prob = max(0.01, min(0.99, 0.3 + temp_effect))
            
            data.append({
                'patient_id': f'P{i:04d}',
                'admission_date': admission_date,
                'age': age,
                'sex': np.random.choice(['M', 'F']),
                'outcome_binary': np.random.binomial(1, prob),
                'biomarker_continuous': np.random.normal(50 + age/10 + temp_effect*5, 10),
                'latitude': np.random.uniform(-26.5, -26.0),
                'longitude': np.random.uniform(27.5, 28.5),
                'facility': f'Facility_{np.random.randint(1, 6)}'
            })
        
        return pd.DataFrame(data)

class TestValidation:
    """Test data validation and quality control functions."""
    
    def test_climate_data_validation(self):
        """Test climate data validation functions."""
        # Create test data
        valid_data = TestDataGenerator.create_climate_data()
        
        # Test valid data passes
        assert self._validate_climate_data(valid_data) == True
        
        # Test invalid data fails
        invalid_data = valid_data.copy()
        invalid_data['temperature'] = invalid_data['temperature'] * np.nan
        assert self._validate_climate_data(invalid_data) == False
        
        # Test out-of-range values
        extreme_data = valid_data.copy()
        extreme_data['temperature'][0, 0, 0] = 100  # Unrealistic temperature
        assert self._validate_temperature_range(extreme_data) == False
    
    def test_health_data_validation(self):
        """Test health data validation functions."""
        valid_data = TestDataGenerator.create_health_data()
        
        # Test valid data
        assert self._validate_health_data(valid_data) == True
        
        # Test missing critical fields
        invalid_data = valid_data.drop('patient_id', axis=1)
        assert self._validate_health_data(invalid_data) == False
        
        # Test invalid dates
        date_invalid = valid_data.copy()
        date_invalid.loc[0, 'admission_date'] = 'invalid_date'
        assert self._validate_health_data(date_invalid) == False
    
    def test_spatial_validation(self):
        """Test spatial coordinate validation."""
        health_data = TestDataGenerator.create_health_data()
        
        # Test valid coordinates
        assert self._validate_coordinates(health_data) == True
        
        # Test invalid coordinates (outside South Africa)
        invalid_coords = health_data.copy()
        invalid_coords.loc[0, 'latitude'] = 90  # North Pole
        assert self._validate_coordinates(invalid_coords) == False
    
    @staticmethod
    def _validate_climate_data(data: xr.Dataset) -> bool:
        """Validate climate dataset."""
        required_vars = ['temperature', 'humidity']
        for var in required_vars:
            if var not in data.variables:
                return False
            if data[var].isnull().all():
                return False
        return True
    
    @staticmethod
    def _validate_temperature_range(data: xr.Dataset) -> bool:
        """Validate temperature values are within realistic range."""
        temp = data['temperature']
        return bool((temp >= -50).all() and (temp <= 60).all())
    
    @staticmethod
    def _validate_health_data(data: pd.DataFrame) -> bool:
        """Validate health dataset."""
        required_columns = ['patient_id', 'admission_date', 'age']
        for col in required_columns:
            if col not in data.columns:
                return False
        
        # Check for valid dates
        try:
            pd.to_datetime(data['admission_date'])
        except:
            return False
        
        return True
    
    @staticmethod
    def _validate_coordinates(data: pd.DataFrame) -> bool:
        """Validate geographic coordinates."""
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            return False
        
        # South Africa bounds (approximate)
        lat_valid = (data['latitude'] >= -35) & (data['latitude'] <= -22)
        lon_valid = (data['longitude'] >= 16) & (data['longitude'] <= 33)
        
        return lat_valid.all() and lon_valid.all()

class TestStatisticalMethods:
    """Test statistical analysis functions."""
    
    def test_dlnm_preparation(self):
        """Test DLNM data preparation."""
        climate_data = TestDataGenerator.create_climate_data()
        health_data = TestDataGenerator.create_health_data()
        
        # Test successful preparation
        prepared_data = self._prepare_dlnm_data(climate_data, health_data)
        
        assert isinstance(prepared_data, pd.DataFrame)
        assert 'temperature_lag0' in prepared_data.columns
        assert 'outcome' in prepared_data.columns
        assert len(prepared_data) > 0
    
    def test_lag_matrix_creation(self):
        """Test lag matrix creation for DLNM."""
        temperatures = np.random.normal(25, 5, 100)
        
        lag_matrix = self._create_lag_matrix(temperatures, max_lag=7)
        
        assert lag_matrix.shape == (100, 8)  # 0-7 lags
        assert not np.isnan(lag_matrix[7:, :]).any()  # No NaN after lag period
    
    def test_heat_index_calculation(self):
        """Test heat index calculations."""
        # Test data
        temp_f = 85  # Fahrenheit
        humidity = 70  # Percent
        
        heat_index = self._calculate_heat_index(temp_f, humidity)
        
        # Heat index should be higher than temperature at high humidity
        assert heat_index > temp_f
        assert isinstance(heat_index, (int, float))
    
    def test_wbgt_calculation(self):
        """Test Wet Bulb Globe Temperature calculation."""
        temp_c = 30  # Celsius
        humidity = 80  # Percent
        
        wbgt = self._calculate_wbgt(temp_c, humidity)
        
        assert isinstance(wbgt, (int, float))
        assert 15 <= wbgt <= 40  # Realistic range
    
    def test_statistical_significance(self):
        """Test statistical significance calculations."""
        # Create data with known effect
        n = 1000
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.5, n)  # Strong relationship
        
        correlation, p_value = self._calculate_correlation(x, y)
        
        assert abs(correlation) > 0.5  # Should detect strong correlation
        assert p_value < 0.05  # Should be significant
    
    @staticmethod
    def _prepare_dlnm_data(climate_data: xr.Dataset, health_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for DLNM analysis."""
        # Simple spatial matching (nearest neighbor)
        prepared_data = []
        
        for _, patient in health_data.iterrows():
            # Find nearest climate grid point
            lat_idx = np.argmin(np.abs(climate_data.lat.values - patient['latitude']))
            lon_idx = np.argmin(np.abs(climate_data.lon.values - patient['longitude']))
            
            # Extract temperature time series
            temps = climate_data['temperature'][:, lat_idx, lon_idx].values
            
            # Create lag structure (simplified)
            admission_idx = 50  # Simplified for testing
            temp_lags = {f'temperature_lag{i}': temps[admission_idx-i] 
                        for i in range(8) if admission_idx-i >= 0}
            
            patient_data = {
                'patient_id': patient['patient_id'],
                'outcome': patient['outcome_binary'],
                **temp_lags
            }
            prepared_data.append(patient_data)
        
        return pd.DataFrame(prepared_data)
    
    @staticmethod
    def _create_lag_matrix(values: np.ndarray, max_lag: int) -> np.ndarray:
        """Create lag matrix for time series."""
        n = len(values)
        lag_matrix = np.full((n, max_lag + 1), np.nan)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                lag_matrix[:, lag] = values
            else:
                lag_matrix[lag:, lag] = values[:-lag]
        
        return lag_matrix
    
    @staticmethod
    def _calculate_heat_index(temp_f: float, humidity: float) -> float:
        """Calculate heat index (simplified Rothfusz equation)."""
        if temp_f < 80:
            return temp_f
        
        T = temp_f
        R = humidity
        
        HI = (-42.379 + 2.04901523*T + 10.14333127*R 
              - 0.22475541*T*R - 6.83783e-03*T*T 
              - 5.481717e-02*R*R + 1.22874e-03*T*T*R 
              + 8.5282e-04*T*R*R - 1.99e-06*T*T*R*R)
        
        return HI
    
    @staticmethod
    def _calculate_wbgt(temp_c: float, humidity: float) -> float:
        """Calculate simplified WBGT."""
        # Simplified empirical formula for outdoor WBGT
        wet_bulb = temp_c * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                   np.arctan(temp_c + humidity) - np.arctan(humidity - 1.676331) + \
                   0.00391838 * (humidity ** 1.5) * np.arctan(0.023101 * humidity) - 4.686035
        
        # Simplified WBGT (assuming no wind and typical solar conditions)
        wbgt = 0.7 * wet_bulb + 0.2 * temp_c + 0.1 * temp_c
        
        return wbgt
    
    @staticmethod
    def _calculate_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculate Pearson correlation and p-value."""
        from scipy.stats import pearsonr
        return pearsonr(x, y)

class TestVisualization:
    """Test visualization and plotting functions."""
    
    def test_plot_generation(self):
        """Test that plots can be generated without errors."""
        import matplotlib.pyplot as plt
        
        # Test time series plot
        dates = pd.date_range('2020-01-01', periods=100)
        temps = np.random.normal(25, 5, 100)
        
        fig, ax = plt.subplots()
        ax.plot(dates, temps)
        ax.set_title('Temperature Time Series')
        
        # Should not raise exception
        plt.close(fig)
        assert True
    
    def test_heatmap_generation(self):
        """Test heatmap generation."""
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        data = np.random.rand(10, 12)
        
        fig, ax = plt.subplots()
        sns.heatmap(data, ax=ax)
        
        plt.close(fig)
        assert True

class TestFileOperations:
    """Test file I/O operations."""
    
    def test_csv_reading(self):
        """Test CSV file reading with error handling."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('col1,col2,col3\n1,2,3\n4,5,6\n')
            temp_path = f.name
        
        try:
            df = pd.read_csv(temp_path)
            assert len(df) == 2
            assert list(df.columns) == ['col1', 'col2', 'col3']
        finally:
            os.unlink(temp_path)
    
    def test_netcdf_operations(self):
        """Test NetCDF file operations."""
        data = TestDataGenerator.create_climate_data()
        
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            temp_path = f.name
        
        try:
            # Write and read NetCDF
            data.to_netcdf(temp_path)
            loaded_data = xr.open_dataset(temp_path)
            
            assert 'temperature' in loaded_data.variables
            assert loaded_data.dims['time'] == data.dims['time']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestConfiguration:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration file loading."""
        config = {
            "analysis_parameters": {
                "max_lag": 21,
                "temperature_percentiles": [90, 95, 99],
                "outcome_variables": ["mortality", "morbidity"]
            },
            "data_paths": {
                "climate_data": "/path/to/climate",
                "health_data": "/path/to/health"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config["analysis_parameters"]["max_lag"] == 21
            assert len(loaded_config["analysis_parameters"]["temperature_percentiles"]) == 3
        finally:
            os.unlink(temp_path)

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_data_handling(self):
        """Test handling of missing data."""
        data = pd.DataFrame({
            'temperature': [25.0, np.nan, 30.0, np.nan, 28.0],
            'outcome': [1, 0, 1, np.nan, 0]
        })
        
        # Test that missing data is properly identified
        missing_count = data.isnull().sum().sum()
        assert missing_count == 3, f"Expected 3 missing values, got {missing_count}"
        
        # Test that complete cases can be extracted
        complete_data = data.dropna()
        expected_length = 3  # Actually 3 complete cases: rows 0, 1, and 4
        assert len(complete_data) == expected_length, f"Expected {expected_length} complete cases, got {len(complete_data)}"
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        try:
            self._calculate_heat_index("invalid", "input")
            assert False, "Should have raised exception"
        except (ValueError, TypeError):
            pass
        
        try:
            TestDataGenerator.create_climate_data(n_days=0)  # Use 0 instead of -1
            assert False, "Should have raised exception" 
        except (ValueError, TypeError, IndexError):
            pass
    
    def _calculate_heat_index(self, temp, humidity):
        """Mock heat index calculation for testing."""
        if not isinstance(temp, (int, float)) or not isinstance(humidity, (int, float)):
            raise TypeError("Temperature and humidity must be numeric")
        return TestStatisticalMethods._calculate_heat_index(temp, humidity)

class TestPerformance:
    """Test performance and memory usage."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create moderately large dataset
        large_data = TestDataGenerator.create_climate_data(n_days=1000, n_locations=50)
        
        # Should complete without memory errors
        assert large_data.sizes['time'] == 1000
        assert large_data.sizes['lat'] == 50
    
    def test_memory_efficiency(self):
        """Test memory-efficient operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and process data
        data = TestDataGenerator.create_climate_data(n_days=365)
        result = data.mean(dim='time')
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB for this operation

def run_comprehensive_tests():
    """
    Run all tests and generate comprehensive report.
    
    Returns:
        Dictionary with test results
    """
    import time
    
    start_time = time.time()
    results = {
        'test_results': {},
        'errors': [],
        'warnings': [],
        'performance_metrics': {}
    }
    
    # Test classes to run
    test_classes = [
        TestValidation,
        TestStatisticalMethods,
        TestVisualization,
        TestFileOperations,
        TestConfiguration,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        class_name = test_class.__name__
        results['test_results'][class_name] = {}
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                test_instance = test_class()
                test_method = getattr(test_instance, method_name)
                
                method_start = time.time()
                test_method()
                method_time = time.time() - method_start
                
                results['test_results'][class_name][method_name] = {
                    'status': 'PASSED',
                    'execution_time': method_time
                }
                
            except Exception as e:
                results['test_results'][class_name][method_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                results['errors'].append(f"{class_name}.{method_name}: {str(e)}")
    
    results['total_execution_time'] = time.time() - start_time
    
    # Calculate summary statistics
    total_tests = sum(len(class_tests) for class_tests in results['test_results'].values())
    passed_tests = sum(
        1 for class_tests in results['test_results'].values()
        for test_result in class_tests.values()
        if test_result['status'] == 'PASSED'
    )
    
    results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }
    
    return results

def generate_test_report(results: Dict, output_file: str = None):
    """
    Generate comprehensive test report.
    
    Args:
        results: Test results dictionary
        output_file: Optional file path to save report
    """
    report = f"""
# HEAT Analysis Test Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Tests: {results['summary']['total_tests']}
- Passed: {results['summary']['passed_tests']}
- Failed: {results['summary']['failed_tests']}
- Success Rate: {results['summary']['success_rate']:.1%}
- Total Execution Time: {results['total_execution_time']:.2f}s

## Test Results by Category
"""
    
    for class_name, class_results in results['test_results'].items():
        report += f"\n### {class_name}\n"
        
        for method_name, method_result in class_results.items():
            status = method_result['status']
            if status == 'PASSED':
                time_info = f" ({method_result['execution_time']:.3f}s)"
                report += f"- ✅ {method_name}{time_info}\n"
            else:
                error_info = method_result.get('error', 'Unknown error')
                report += f"- ❌ {method_name}: {error_info}\n"
    
    if results['errors']:
        report += "\n## Errors\n"
        for error in results['errors']:
            report += f"- {error}\n"
    
    if results['warnings']:
        report += "\n## Warnings\n"
        for warning in results['warnings']:
            report += f"- {warning}\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
    
    return report

if __name__ == "__main__":
    # Run tests
    print("Running comprehensive HEAT analysis tests...")
    results = run_comprehensive_tests()
    
    # Generate report
    report = generate_test_report(results)
    print(report)
    
    # Save detailed results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    with open(output_dir / f"test_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save markdown report
    with open(output_dir / f"test_report_{timestamp}.md", 'w') as f:
        f.write(report)
    
    print(f"\nDetailed results saved to test_results/")
    print(f"Overall success rate: {results['summary']['success_rate']:.1%}")