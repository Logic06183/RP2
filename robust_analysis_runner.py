#!/usr/bin/env python3
"""
Robust Analysis Runner for HEAT Climate-Health Research
Integrates archival system, comprehensive testing, and reproducibility measures.
"""

import os
import sys
import json
import logging
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from archive_manager import AnalysisArchiveManager
from test_framework import run_comprehensive_tests, generate_test_report

class RobustAnalysisRunner:
    """
    Manages the complete analysis lifecycle with built-in testing, archival, and reproducibility.
    
    Features:
    - Pre-analysis validation and testing
    - Automatic archival of previous results
    - Environment and dependency verification
    - Error handling with detailed logging
    - Post-analysis validation
    - Reproducibility measures
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the robust analysis runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.base_dir = Path.cwd()
        self.config_path = Path(config_path) if config_path else self.base_dir / "analysis_config.json"
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.archive_manager = AnalysisArchiveManager(str(self.base_dir))
        
        # Analysis state tracking
        self.analysis_state = {
            'start_time': None,
            'end_time': None,
            'status': 'not_started',
            'archived_version': None,
            'test_results': None,
            'errors': [],
            'warnings': []
        }
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"analysis_run_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Analysis runner initialized. Log file: {log_file}")
    
    def _load_configuration(self) -> Dict:
        """Load analysis configuration."""
        default_config = {
            "analysis_scripts": [
                "publication_ready_dlnm_analysis.py",
                "src/comprehensive_final_analysis.py"
            ],
            "test_before_analysis": True,
            "archive_before_analysis": True,
            "validate_environment": True,
            "max_execution_time": 3600,  # 1 hour
            "required_packages": [
                "numpy", "pandas", "xarray", "matplotlib", "seaborn",
                "scipy", "sklearn", "statsmodels"
            ],
            "output_validation": {
                "required_files": [
                    "results/dlnm_results.json",
                    "results/analysis_summary.csv"
                ],
                "min_file_sizes": {
                    "results/dlnm_results.json": 1000,  # bytes
                    "results/analysis_summary.csv": 500
                }
            },
            "notification": {
                "email_on_completion": False,
                "email_on_error": False
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                # For nested dicts, we need deep merge
                merged_config = default_config.copy()
                for key, value in config.items():
                    if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                return merged_config
        else:
            # Save default configuration
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Created default configuration at {self.config_path}")
            return default_config
    
    def _validate_environment(self) -> bool:
        """
        Validate analysis environment and dependencies.
        
        Returns:
            True if environment is valid, False otherwise
        """
        self.logger.info("Validating analysis environment...")
        
        validation_errors = []
        
        # Check Python version
        if sys.version_info < (3, 7):
            validation_errors.append(f"Python 3.7+ required, found {sys.version}")
        
        # Check required packages
        missing_packages = []
        for package in self.config["required_packages"]:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            validation_errors.append(f"Missing packages: {', '.join(missing_packages)}")
        
        # Check disk space (at least 1GB free)
        stat = os.statvfs(self.base_dir)
        # Use f_bavail if f_available is not available (compatibility)
        available_blocks = getattr(stat, 'f_available', stat.f_bavail)
        free_space_gb = (stat.f_frsize * available_blocks) / (1024**3)
        if free_space_gb < 1.0:
            validation_errors.append(f"Low disk space: {free_space_gb:.2f}GB free")
        
        # Check write permissions
        test_file = self.base_dir / "test_write_permissions.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            validation_errors.append(f"No write permissions: {e}")
        
        if validation_errors:
            self.logger.error("Environment validation failed:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            return False
        
        self.logger.info("Environment validation passed")
        return True
    
    def _run_pre_analysis_tests(self) -> bool:
        """
        Run comprehensive tests before analysis.
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("Running pre-analysis tests...")
        
        try:
            test_results = run_comprehensive_tests()
            self.analysis_state['test_results'] = test_results
            
            success_rate = test_results['summary']['success_rate']
            self.logger.info(f"Pre-analysis tests completed. Success rate: {success_rate:.1%}")
            
            if success_rate < 0.8:  # 80% threshold
                self.logger.error("Pre-analysis tests failed. Success rate below threshold.")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-analysis testing failed: {e}")
            self.analysis_state['errors'].append(f"Pre-analysis testing: {str(e)}")
            return False
    
    def _archive_previous_analysis(self) -> Optional[str]:
        """
        Archive previous analysis results.
        
        Returns:
            Path to created archive or None if failed
        """
        if not self.config["archive_before_analysis"]:
            return None
        
        self.logger.info("Archiving previous analysis...")
        
        try:
            # Check if there are results to archive
            results_dir = self.base_dir / "results"
            if not results_dir.exists() or not any(results_dir.iterdir()):
                self.logger.info("No previous results to archive")
                return None
            
            # Create archive with timestamp description
            description = f"Pre_analysis_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            archive_path = self.archive_manager.archive_current_analysis(
                version_type="minor",
                description=description
            )
            
            self.analysis_state['archived_version'] = archive_path
            self.logger.info(f"Previous analysis archived to: {archive_path}")
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Archival failed: {e}")
            self.analysis_state['warnings'].append(f"Archival failed: {str(e)}")
            return None
    
    def _execute_analysis_script(self, script_path: str) -> bool:
        """
        Execute individual analysis script with monitoring.
        
        Args:
            script_path: Path to analysis script
            
        Returns:
            True if successful, False otherwise
        """
        script_file = self.base_dir / script_path
        if not script_file.exists():
            self.logger.error(f"Analysis script not found: {script_path}")
            return False
        
        self.logger.info(f"Executing analysis script: {script_path}")
        
        try:
            # Execute script with timeout
            result = subprocess.run(
                [sys.executable, str(script_file)],
                cwd=str(self.base_dir),
                timeout=self.config["max_execution_time"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Script completed successfully: {script_path}")
                if result.stdout:
                    self.logger.debug(f"Script output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Script failed with return code {result.returncode}: {script_path}")
                if result.stderr:
                    self.logger.error(f"Script error: {result.stderr}")
                self.analysis_state['errors'].append(f"{script_path}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Script timeout after {self.config['max_execution_time']}s: {script_path}")
            self.analysis_state['errors'].append(f"{script_path}: Timeout")
            return False
        except Exception as e:
            self.logger.error(f"Script execution failed: {script_path} - {e}")
            self.analysis_state['errors'].append(f"{script_path}: {str(e)}")
            return False
    
    def _validate_analysis_outputs(self) -> bool:
        """
        Validate analysis outputs against expected requirements.
        
        Returns:
            True if validation passes, False otherwise
        """
        self.logger.info("Validating analysis outputs...")
        
        validation_errors = []
        output_config = self.config["output_validation"]
        
        # Check required files exist
        for required_file in output_config.get("required_files", []):
            file_path = self.base_dir / required_file
            if not file_path.exists():
                validation_errors.append(f"Missing required output file: {required_file}")
            else:
                # Check minimum file size if specified
                min_sizes = output_config.get("min_file_sizes", {})
                if required_file in min_sizes:
                    actual_size = file_path.stat().st_size
                    min_size = min_sizes[required_file]
                    if actual_size < min_size:
                        validation_errors.append(
                            f"Output file too small: {required_file} "
                            f"({actual_size} bytes < {min_size} bytes)"
                        )
        
        # Additional validation checks
        results_dir = self.base_dir / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*.json"))
            if not result_files:
                validation_errors.append("No JSON result files found in results directory")
        
        if validation_errors:
            self.logger.error("Output validation failed:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            return False
        
        self.logger.info("Output validation passed")
        return True
    
    def _generate_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report.
        
        Returns:
            Report content as string
        """
        report = f"""
# HEAT Analysis Run Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Execution Summary
- Start Time: {self.analysis_state['start_time']}
- End Time: {self.analysis_state['end_time']}
- Duration: {self._calculate_duration()}
- Status: {self.analysis_state['status'].upper()}
- Archived Version: {self.analysis_state['archived_version'] or 'None'}

## Configuration
- Scripts Executed: {len(self.config['analysis_scripts'])}
- Testing Enabled: {self.config['test_before_analysis']}
- Archival Enabled: {self.config['archive_before_analysis']}
- Environment Validation: {self.config['validate_environment']}

## Analysis Scripts
"""
        
        for script in self.config['analysis_scripts']:
            status = "✅ EXECUTED" if self.analysis_state['status'] == 'completed' else "❌ FAILED"
            report += f"- {script}: {status}\n"
        
        if self.analysis_state['test_results']:
            test_summary = self.analysis_state['test_results']['summary']
            report += f"""
## Test Results
- Total Tests: {test_summary['total_tests']}
- Passed: {test_summary['passed_tests']}
- Failed: {test_summary['failed_tests']}
- Success Rate: {test_summary['success_rate']:.1%}
"""
        
        if self.analysis_state['errors']:
            report += "\n## Errors\n"
            for error in self.analysis_state['errors']:
                report += f"- {error}\n"
        
        if self.analysis_state['warnings']:
            report += "\n## Warnings\n"
            for warning in self.analysis_state['warnings']:
                report += f"- {warning}\n"
        
        return report
    
    def _calculate_duration(self) -> str:
        """Calculate analysis duration."""
        if not self.analysis_state['start_time'] or not self.analysis_state['end_time']:
            return "Unknown"
        
        duration = self.analysis_state['end_time'] - self.analysis_state['start_time']
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    
    def run_analysis(self, skip_tests: bool = False, skip_archival: bool = False) -> bool:
        """
        Run complete analysis pipeline with robustness measures.
        
        Args:
            skip_tests: Skip pre-analysis testing
            skip_archival: Skip archival of previous results
            
        Returns:
            True if analysis completed successfully, False otherwise
        """
        self.analysis_state['start_time'] = datetime.now()
        self.analysis_state['status'] = 'running'
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING ROBUST HEAT ANALYSIS PIPELINE")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Environment validation
            if self.config["validate_environment"]:
                if not self._validate_environment():
                    self.analysis_state['status'] = 'failed_validation'
                    return False
            
            # Step 2: Pre-analysis testing
            if self.config["test_before_analysis"] and not skip_tests:
                if not self._run_pre_analysis_tests():
                    self.analysis_state['status'] = 'failed_testing'
                    return False
            
            # Step 3: Archive previous results
            if not skip_archival:
                self._archive_previous_analysis()
            
            # Step 4: Execute analysis scripts
            all_scripts_successful = True
            for script in self.config["analysis_scripts"]:
                if not self._execute_analysis_script(script):
                    all_scripts_successful = False
                    break  # Stop on first failure
            
            if not all_scripts_successful:
                self.analysis_state['status'] = 'failed_execution'
                return False
            
            # Step 5: Validate outputs
            if not self._validate_analysis_outputs():
                self.analysis_state['status'] = 'failed_output_validation'
                return False
            
            # Step 6: Success!
            self.analysis_state['status'] = 'completed'
            self.analysis_state['end_time'] = datetime.now()
            
            # Generate final report
            report = self._generate_analysis_report()
            report_file = self.base_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            self.logger.info("=" * 60)
            self.logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Report saved to: {report_file}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.analysis_state['status'] = 'failed_error'
            self.analysis_state['end_time'] = datetime.now()
            self.analysis_state['errors'].append(f"Pipeline error: {str(e)}")
            
            self.logger.error("=" * 60)
            self.logger.error("ANALYSIS PIPELINE FAILED!")
            self.logger.error(f"Error: {e}")
            self.logger.error("Traceback:")
            self.logger.error(traceback.format_exc())
            self.logger.error("=" * 60)
            
            return False
    
    def status(self) -> Dict:
        """Return current analysis status."""
        return self.analysis_state.copy()

def main():
    """Command line interface for robust analysis runner."""
    parser = argparse.ArgumentParser(description="Robust HEAT Analysis Runner")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pre-analysis testing")
    parser.add_argument("--skip-archival", action="store_true", help="Skip archival of previous results")
    parser.add_argument("--status-only", action="store_true", help="Show status only")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = RobustAnalysisRunner(args.config)
    
    if args.status_only:
        status = runner.status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Run analysis
    success = runner.run_analysis(
        skip_tests=args.skip_tests,
        skip_archival=args.skip_archival
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()