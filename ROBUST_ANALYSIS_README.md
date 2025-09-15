# Robust HEAT Analysis System

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
