#!/usr/bin/env python3
import sys
from pathlib import Path

def validate_setup():
    """Validate that all components are properly installed."""
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
            print(f"   - {file_name}")
        return False
    
    print("✅ All required files present")
    
    # Test imports
    try:
        from archive_manager import AnalysisArchiveManager
        from test_framework import run_comprehensive_tests
        from reproducibility_manager import ReproducibilityManager
        print("✅ All modules can be imported")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("🎉 Robust Analysis System setup is valid!")
    return True

if __name__ == "__main__":
    success = validate_setup()
    sys.exit(0 if success else 1)
