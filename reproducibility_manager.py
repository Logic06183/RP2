#!/usr/bin/env python3
"""
Reproducibility Manager for HEAT Climate-Health Analysis
Ensures complete reproducibility through environment tracking, seed management, and provenance.
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
import pkg_resources
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

class ReproducibilityManager:
    """
    Manages all aspects of analysis reproducibility including:
    - Environment capture and restoration
    - Random seed management
    - Data provenance tracking
    - Dependency versioning
    - Configuration management
    """
    
    def __init__(self, project_dir: str = None):
        """
        Initialize reproducibility manager.
        
        Args:
            project_dir: Project directory (defaults to current directory)
        """
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.repro_dir = self.project_dir / "reproducibility"
        self.repro_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.repro_dir / "environments").mkdir(exist_ok=True)
        (self.repro_dir / "seeds").mkdir(exist_ok=True)
        (self.repro_dir / "provenance").mkdir(exist_ok=True)
        (self.repro_dir / "checksums").mkdir(exist_ok=True)
        
        self.current_session = {
            'session_id': self._generate_session_id(),
            'timestamp': datetime.now().isoformat(),
            'environment': None,
            'seeds': {},
            'data_checksums': {},
            'config_hash': None
        }
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_suffix}"
    
    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture complete computational environment.
        
        Returns:
            Dictionary containing environment information
        """
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session['session_id'],
            'system': {
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            },
            'environment_variables': {
                key: value for key, value in os.environ.items()
                if key.startswith(('CONDA_', 'VIRTUAL_ENV', 'PATH', 'PYTHONPATH'))
            },
            'working_directory': str(self.project_dir),
            'python_packages': self._get_package_versions(),
            'system_packages': self._get_system_packages(),
            'hardware': self._get_hardware_info(),
            'git_info': self._get_git_info()
        }
        
        # Save environment
        env_file = self.repro_dir / "environments" / f"{self.current_session['session_id']}_environment.json"
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
        
        self.current_session['environment'] = env_info
        return env_info
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of all installed Python packages."""
        packages = {}
        
        # Get all installed packages
        try:
            installed_packages = [d for d in pkg_resources.working_set]
            for package in installed_packages:
                packages[package.project_name] = package.version
        except Exception:
            # Fallback method
            try:
                result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if '==' in line:
                        name, version = line.split('==')
                        packages[name] = version
            except Exception:
                packages['error'] = 'Could not retrieve package versions'
        
        return packages
    
    def _get_system_packages(self) -> Dict[str, Any]:
        """Get system-level package information."""
        system_info = {}
        
        # Check for conda environment
        if 'CONDA_DEFAULT_ENV' in os.environ:
            system_info['conda_env'] = os.environ['CONDA_DEFAULT_ENV']
            try:
                result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
                system_info['conda_packages'] = result.stdout
            except Exception:
                system_info['conda_packages'] = 'Could not retrieve conda packages'
        
        # Check for virtual environment
        if 'VIRTUAL_ENV' in os.environ:
            system_info['virtual_env'] = os.environ['VIRTUAL_ENV']
        
        return system_info
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        import psutil
        
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free
            }
        }
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information."""
        try:
            git_info = {
                'commit_hash': subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                                     cwd=self.project_dir).decode().strip(),
                'branch': subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                                cwd=self.project_dir).decode().strip(),
                'is_dirty': len(subprocess.check_output(['git', 'status', '--porcelain'], 
                                                       cwd=self.project_dir)) > 0,
                'remote_url': subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], 
                                                    cwd=self.project_dir).decode().strip(),
                'last_commit_date': subprocess.check_output(['git', 'log', '-1', '--format=%cd'], 
                                                           cwd=self.project_dir).decode().strip()
            }
        except Exception:
            git_info = {'error': 'Not a git repository or git not available'}
        
        return git_info
    
    def set_global_seeds(self, seed: int = None) -> Dict[str, int]:
        """
        Set seeds for all random number generators.
        
        Args:
            seed: Master seed (generated if None)
            
        Returns:
            Dictionary of seeds used for each library
        """
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        seeds = {'master_seed': seed}
        
        # Set NumPy seed
        np.random.seed(seed)
        seeds['numpy'] = seed
        
        # Set Python random seed
        import random
        random.seed(seed)
        seeds['python_random'] = seed
        
        # Set pandas seed (if applicable)
        try:
            import pandas as pd
            # Pandas uses numpy random state
            seeds['pandas'] = seed
        except ImportError:
            pass
        
        # Set scikit-learn seed
        try:
            from sklearn.utils import check_random_state
            seeds['sklearn'] = seed
        except ImportError:
            pass
        
        # Set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            seeds['tensorflow'] = seed
        except ImportError:
            pass
        
        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            seeds['pytorch'] = seed
        except ImportError:
            pass
        
        # Save seeds
        seed_file = self.repro_dir / "seeds" / f"{self.current_session['session_id']}_seeds.json"
        with open(seed_file, 'w') as f:
            json.dump(seeds, f, indent=2)
        
        self.current_session['seeds'] = seeds
        return seeds
    
    def calculate_data_checksums(self, data_files: List[str] = None) -> Dict[str, str]:
        """
        Calculate checksums for data files to ensure data integrity.
        
        Args:
            data_files: List of data files to check (auto-detect if None)
            
        Returns:
            Dictionary mapping file paths to their checksums
        """
        if data_files is None:
            # Auto-detect data files
            data_patterns = ['*.csv', '*.json', '*.nc', '*.h5', '*.parquet']
            data_files = []
            for pattern in data_patterns:
                data_files.extend(self.project_dir.glob(f"**/{pattern}"))
            data_files = [str(f) for f in data_files]
        
        checksums = {}
        
        for file_path in data_files:
            file_obj = Path(file_path)
            if file_obj.exists():
                # Calculate SHA-256 checksum
                sha256_hash = hashlib.sha256()
                with open(file_obj, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                
                checksums[str(file_obj.relative_to(self.project_dir))] = sha256_hash.hexdigest()
        
        # Save checksums
        checksum_file = self.repro_dir / "checksums" / f"{self.current_session['session_id']}_checksums.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        
        self.current_session['data_checksums'] = checksums
        return checksums
    
    def create_provenance_record(self, 
                                analysis_name: str,
                                input_files: List[str] = None,
                                output_files: List[str] = None,
                                parameters: Dict = None,
                                description: str = None) -> Dict[str, Any]:
        """
        Create complete provenance record for analysis.
        
        Args:
            analysis_name: Name of the analysis
            input_files: List of input files
            output_files: List of output files
            parameters: Analysis parameters
            description: Analysis description
            
        Returns:
            Provenance record dictionary
        """
        provenance = {
            'analysis_name': analysis_name,
            'session_id': self.current_session['session_id'],
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'input_files': input_files or [],
            'output_files': output_files or [],
            'parameters': parameters or {},
            'environment': self.current_session.get('environment'),
            'seeds': self.current_session.get('seeds'),
            'data_checksums': self.current_session.get('data_checksums', {}),
            'execution_context': {
                'command_line': ' '.join(sys.argv),
                'script_path': sys.argv[0] if sys.argv else None,
                'user': os.getenv('USER', 'unknown'),
                'hostname': platform.node()
            }
        }
        
        # Save provenance record
        provenance_file = self.repro_dir / "provenance" / f"{analysis_name}_{self.current_session['session_id']}.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        
        return provenance
    
    def generate_requirements_file(self, output_file: str = None) -> str:
        """
        Generate requirements.txt file for current environment.
        
        Args:
            output_file: Output file path (defaults to requirements.txt)
            
        Returns:
            Path to generated requirements file
        """
        if output_file is None:
            output_file = self.project_dir / "requirements.txt"
        
        # Get package versions
        packages = self._get_package_versions()
        
        # Write requirements file
        with open(output_file, 'w') as f:
            f.write(f"# Requirements file generated by ReproducibilityManager\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Session ID: {self.current_session['session_id']}\n\n")
            
            for package, version in sorted(packages.items()):
                if package != 'error':
                    f.write(f"{package}=={version}\n")
        
        return str(output_file)
    
    def generate_conda_environment_file(self, output_file: str = None) -> str:
        """
        Generate conda environment.yml file.
        
        Args:
            output_file: Output file path (defaults to environment.yml)
            
        Returns:
            Path to generated environment file
        """
        if output_file is None:
            output_file = self.project_dir / "environment.yml"
        
        # Get environment info
        env_info = self.current_session.get('environment', {})
        packages = env_info.get('python_packages', {})
        
        # Create environment specification
        env_spec = {
            'name': 'heat-analysis',
            'channels': ['conda-forge', 'defaults'],
            'dependencies': [
                f"python={env_info.get('system', {}).get('python_version', '3.8')}",
            ]
        }
        
        # Add major scientific packages
        major_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 
                         'xarray', 'netcdf4', 'scikit-learn', 'statsmodels']
        
        pip_packages = []
        for package, version in packages.items():
            if package in major_packages:
                env_spec['dependencies'].append(f"{package}={version}")
            else:
                pip_packages.append(f"{package}=={version}")
        
        if pip_packages:
            env_spec['dependencies'].append({'pip': pip_packages})
        
        # Write YAML file
        import yaml
        with open(output_file, 'w') as f:
            f.write(f"# Conda environment file generated by ReproducibilityManager\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Session ID: {self.current_session['session_id']}\n\n")
            yaml.dump(env_spec, f, default_flow_style=False)
        
        return str(output_file)
    
    def verify_reproducibility(self, 
                             reference_session: str = None,
                             reference_file: str = None) -> Dict[str, Any]:
        """
        Verify reproducibility against reference environment.
        
        Args:
            reference_session: Reference session ID
            reference_file: Reference environment file
            
        Returns:
            Verification results
        """
        if reference_file:
            with open(reference_file, 'r') as f:
                reference_env = json.load(f)
        elif reference_session:
            ref_file = self.repro_dir / "environments" / f"{reference_session}_environment.json"
            with open(ref_file, 'r') as f:
                reference_env = json.load(f)
        else:
            raise ValueError("Must provide either reference_session or reference_file")
        
        current_env = self.current_session.get('environment', {})
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'reference_session': reference_session,
            'current_session': self.current_session['session_id'],
            'differences': {},
            'warnings': [],
            'errors': []
        }
        
        # Check Python version
        ref_python = reference_env.get('system', {}).get('python_version')
        cur_python = current_env.get('system', {}).get('python_version')
        if ref_python != cur_python:
            verification['differences']['python_version'] = {
                'reference': ref_python,
                'current': cur_python
            }
        
        # Check package versions
        ref_packages = reference_env.get('python_packages', {})
        cur_packages = current_env.get('python_packages', {})
        
        package_diffs = {}
        for package, ref_version in ref_packages.items():
            cur_version = cur_packages.get(package)
            if cur_version != ref_version:
                package_diffs[package] = {
                    'reference': ref_version,
                    'current': cur_version or 'MISSING'
                }
        
        if package_diffs:
            verification['differences']['packages'] = package_diffs
        
        # Check for critical differences
        critical_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'xarray']
        for pkg in critical_packages:
            if pkg in package_diffs:
                verification['warnings'].append(f"Critical package version mismatch: {pkg}")
        
        return verification
    
    def create_reproducibility_report(self) -> str:
        """
        Create comprehensive reproducibility report.
        
        Returns:
            Report content as string
        """
        env_info = self.current_session.get('environment', {})
        
        report = f"""
# Reproducibility Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.current_session['session_id']}

## Environment Information
- Platform: {env_info.get('system', {}).get('platform', 'Unknown')}
- Python Version: {env_info.get('system', {}).get('python_version', 'Unknown')}
- Working Directory: {env_info.get('working_directory', 'Unknown')}

## Random Seeds
"""
        
        seeds = self.current_session.get('seeds', {})
        if seeds:
            for lib, seed in seeds.items():
                report += f"- {lib}: {seed}\n"
        else:
            report += "- No seeds recorded\n"
        
        report += "\n## Data Integrity\n"
        checksums = self.current_session.get('data_checksums', {})
        if checksums:
            report += f"- {len(checksums)} data files checksummed\n"
            for file_path, checksum in checksums.items():
                report += f"  - {file_path}: {checksum[:16]}...\n"
        else:
            report += "- No data checksums recorded\n"
        
        report += f"\n## Package Versions\n"
        packages = env_info.get('python_packages', {})
        critical_packages = ['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 
                           'xarray', 'scikit-learn', 'statsmodels']
        
        for pkg in critical_packages:
            version = packages.get(pkg, 'Not installed')
            report += f"- {pkg}: {version}\n"
        
        return report

def create_analysis_config(output_file: str = None) -> str:
    """
    Create analysis configuration template.
    
    Args:
        output_file: Output file path
        
    Returns:
        Path to created configuration file
    """
    if output_file is None:
        output_file = "analysis_config.json"
    
    config = {
        "analysis_metadata": {
            "title": "HEAT Climate-Health Analysis",
            "version": "1.0.0",
            "description": "Distributed lag non-linear model analysis of heat exposure and health outcomes",
            "authors": ["HEAT Research Team"],
            "contact": "heat-research@example.com",
            "keywords": ["climate", "health", "heat", "DLNM", "Africa"]
        },
        "reproducibility": {
            "set_seeds": True,
            "master_seed": 42,
            "capture_environment": True,
            "calculate_checksums": True,
            "track_provenance": True
        },
        "analysis_parameters": {
            "dlnm": {
                "max_lag": 21,
                "temperature_percentiles": [75, 90, 95, 99],
                "df_temperature": 4,
                "df_lag": 4,
                "model_family": "binomial"
            },
            "spatial": {
                "buffer_distance_km": 10,
                "interpolation_method": "nearest",
                "coordinate_system": "WGS84"
            },
            "temporal": {
                "date_format": "%Y-%m-%d",
                "time_window_days": 30,
                "aggregation_method": "mean"
            }
        },
        "data_sources": {
            "climate_data": {
                "path": "data/climate/",
                "format": "netcdf",
                "variables": ["temperature", "humidity", "pressure"],
                "quality_checks": True
            },
            "health_data": {
                "path": "data/health/",
                "format": "csv",
                "key_variables": ["patient_id", "admission_date", "outcome"],
                "privacy_level": "high"
            }
        },
        "output_specifications": {
            "results_directory": "results/",
            "figure_directory": "figures/",
            "table_directory": "tables/",
            "formats": ["json", "csv", "png", "pdf"],
            "dpi": 300,
            "figure_size": [12, 8]
        },
        "quality_control": {
            "run_tests": True,
            "validate_inputs": True,
            "check_outputs": True,
            "generate_report": True
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_file

if __name__ == "__main__":
    # Demo usage
    manager = ReproducibilityManager()
    
    print("Capturing environment...")
    env = manager.capture_environment()
    
    print("Setting global seeds...")
    seeds = manager.set_global_seeds(42)
    
    print("Calculating data checksums...")
    checksums = manager.calculate_data_checksums()
    
    print("Generating reproducibility report...")
    report = manager.create_reproducibility_report()
    
    # Save report
    with open("reproducibility_report.md", 'w') as f:
        f.write(report)
    
    print("Creating requirements files...")
    manager.generate_requirements_file()
    manager.generate_conda_environment_file()
    
    print("Creating analysis configuration...")
    create_analysis_config()
    
    print("Reproducibility setup complete!")
    print(f"Session ID: {manager.current_session['session_id']}")
    print(f"Files created in: {manager.repro_dir}")