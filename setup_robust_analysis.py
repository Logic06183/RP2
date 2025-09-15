#!/usr/bin/env python3
"""
Setup script for Robust HEAT Analysis System
Initializes the complete analysis infrastructure with archival, testing, and reproducibility.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def setup_robust_analysis(project_dir: str = None):
    """
    Setup complete robust analysis infrastructure.
    
    Args:
        project_dir: Project directory (defaults to current)
    """
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir)
    
    print("🔧 Setting up Robust HEAT Analysis System")
    print(f"📁 Project directory: {project_dir}")
    
    # Create necessary directories
    directories = [
        "archives",
        "current",
        "logs", 
        "test_results",
        "reproducibility",
        "reproducibility/environments",
        "reproducibility/seeds",
        "reproducibility/provenance",
        "reproducibility/checksums",
        "tests",
        "configs"
    ]
    
    for dir_name in directories:
        dir_path = project_dir / dir_name
        dir_path.mkdir(exist_ok=True, parents=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Create configuration files
    configs = {
        "analysis_config.json": {
            "analysis_scripts": [
                "publication_ready_dlnm_analysis.py",
                "src/comprehensive_final_analysis.py"
            ],
            "test_before_analysis": True,
            "archive_before_analysis": True,
            "validate_environment": True,
            "max_execution_time": 3600,
            "required_packages": [
                "numpy", "pandas", "xarray", "matplotlib", "seaborn",
                "scipy", "scikit-learn", "statsmodels", "netcdf4"
            ],
            "output_validation": {
                "required_files": [
                    "results/dlnm_results.json",
                    "results/analysis_summary.csv"
                ],
                "min_file_sizes": {
                    "results/dlnm_results.json": 1000,
                    "results/analysis_summary.csv": 500
                }
            }
        },
        "archive_config.json": {
            "version": "1.0.0",
            "archive_retention_days": 365,
            "auto_archive_enabled": True,
            "files_to_track": [
                "*.py", "*.R", "*.ipynb",
                "results/*.csv", "results/*.json", "results/*.png", "results/*.pdf",
                "tables/*.csv", "tables/*.tex",
                "figures/*.png", "figures/*.pdf"
            ],
            "exclude_patterns": [
                "__pycache__", ".pytest_cache", "*.pyc", ".DS_Store",
                "logs/*", "test_results/*", "archives/*"
            ]
        }
    }
    
    for config_file, config_data in configs.items():
        config_path = project_dir / config_file
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"✅ Created configuration: {config_file}")
    
    # Create executable shell scripts
    shell_scripts = {
        "run_analysis.sh": """#!/bin/bash
# Robust HEAT Analysis Runner
echo "🔥 Starting Robust HEAT Analysis Pipeline"
python robust_analysis_runner.py "$@"
""",
        "run_tests.sh": """#!/bin/bash
# Run comprehensive tests
echo "🧪 Running comprehensive tests"
python test_framework.py
""",
        "archive_results.sh": """#!/bin/bash
# Archive current results
echo "📦 Archiving current results"
python archive_manager.py archive --description "Manual_archive_$(date +%Y%m%d_%H%M%S)"
""",
        "setup_reproducibility.sh": """#!/bin/bash
# Setup reproducibility environment
echo "🔬 Setting up reproducibility environment"
python reproducibility_manager.py
"""
    }
    
    for script_name, script_content in shell_scripts.items():
        script_path = project_dir / script_name
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)  # Make executable
        print(f"✅ Created executable script: {script_name}")
    
    # Create README for the robust analysis system
    readme_content = """# Robust HEAT Analysis System

This directory contains a complete robust analysis infrastructure with automated archival, comprehensive testing, and full reproducibility tracking.

## Quick Start

### Run Complete Analysis Pipeline
```bash
./run_analysis.sh
```

### Run Tests Only
```bash
./run_tests.sh
```

### Archive Current Results
```bash
./archive_results.sh
```

### Setup Reproducibility
```bash
./setup_reproducibility.sh
```

## System Components

### 1. Analysis Runner (`robust_analysis_runner.py`)
- Orchestrates complete analysis pipeline
- Handles pre-analysis validation and testing
- Manages archival of previous results
- Validates outputs and generates reports

### 2. Archive Manager (`archive_manager.py`)
- Automatic versioning with semantic versioning
- Preserves analysis history with full metadata
- Enables comparison between analysis versions
- Maintains configurable retention policies

### 3. Test Framework (`test_framework.py`)
- Comprehensive unit tests for all analysis components
- Data validation and quality control
- Statistical method verification
- Performance and memory usage testing

### 4. Reproducibility Manager (`reproducibility_manager.py`)
- Complete environment capture and restoration
- Random seed management across all libraries
- Data provenance tracking with checksums
- Dependency versioning and requirements generation

## Directory Structure

```
├── archives/                    # Archived analysis versions
├── current/                     # Current analysis outputs
├── logs/                       # Execution logs
├── test_results/               # Test outputs and reports
├── reproducibility/            # Reproducibility artifacts
│   ├── environments/          # Environment snapshots
│   ├── seeds/                 # Random seed records
│   ├── provenance/            # Analysis provenance
│   └── checksums/             # Data integrity checksums
├── tests/                      # Additional test files
└── configs/                    # Configuration files

```

## Configuration

### Analysis Configuration (`analysis_config.json`)
- Analysis scripts to execute
- Testing and archival settings
- Environment validation requirements
- Output validation criteria

### Archive Configuration (`archive_config.json`)
- File tracking patterns
- Retention policies
- Versioning settings

## Best Practices

1. **Always run tests** before conducting analysis
2. **Archive regularly** to preserve analysis history
3. **Use consistent seeds** for reproducible results
4. **Validate outputs** after each analysis run
5. **Document changes** in archive descriptions

## Troubleshooting

### Environment Issues
```bash
python robust_analysis_runner.py --status-only
```

### Test Failures
Check `test_results/` for detailed test reports

### Archive Problems
```bash
python archive_manager.py list
```

### Reproducibility Issues
```bash
python reproducibility_manager.py
```

## Advanced Usage

### Custom Analysis Scripts
Add scripts to `analysis_scripts` in `analysis_config.json`

### Custom Tests
Add test classes to `test_framework.py`

### Environment Restoration
```bash
# Generate requirements
python reproducibility_manager.py

# Create conda environment
conda env create -f environment.yml
```

## Support

For issues or questions, check the logs in `logs/` directory or run with verbose output:
```bash
python robust_analysis_runner.py --verbose
```
"""
    
    readme_path = project_dir / "ROBUST_ANALYSIS_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✅ Created documentation: ROBUST_ANALYSIS_README.md")
    
    # Create a simple validation script
    validation_script = f"""#!/usr/bin/env python3
import sys
from pathlib import Path

def validate_setup():
    \"\"\"Validate that all components are properly installed.\"\"\"
    project_dir = Path.cwd()
    
    # Check required files
    required_files = [
        "robust_analysis_runner.py",
        "archive_manager.py", 
        "test_framework.py",
        "reproducibility_manager.py",
        "analysis_config.json",
        "archive_config.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (project_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_name in missing_files:
            print(f"   - {{file_name}}")
        return False
    
    print("✅ All required files present")
    
    # Test imports
    try:
        from archive_manager import AnalysisArchiveManager
        from test_framework import run_comprehensive_tests
        from reproducibility_manager import ReproducibilityManager
        print("✅ All modules can be imported")
    except Exception as e:
        print(f"❌ Import error: {{e}}")
        return False
    
    print("🎉 Robust Analysis System setup is valid!")
    return True

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
"""
    
    validation_path = project_dir / "validate_setup.py"
    with open(validation_path, 'w') as f:
        f.write(validation_script)
    os.chmod(validation_path, 0o755)
    print(f"✅ Created validation script: validate_setup.py")
    
    print("\n🎉 Robust HEAT Analysis System setup complete!")
    print("\nNext steps:")
    print("1. Run validation: python validate_setup.py")
    print("2. Setup reproducibility: ./setup_reproducibility.sh")
    print("3. Run initial tests: ./run_tests.sh")
    print("4. Execute analysis: ./run_analysis.sh")
    print(f"\n📖 See ROBUST_ANALYSIS_README.md for detailed documentation")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup Robust HEAT Analysis System")
    parser.add_argument("--project-dir", help="Project directory (defaults to current)")
    
    args = parser.parse_args()
    setup_robust_analysis(args.project_dir)