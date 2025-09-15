#!/usr/bin/env python3
"""
HEAT Analysis Archive Manager
Implements automated archival system for maintaining analysis versions and ensuring reproducibility.
"""

import os
import shutil
import datetime
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import git

class AnalysisArchiveManager:
    """
    Manages archival of analysis runs with semantic versioning and provenance tracking.
    
    Features:
    - Automatic archival of previous analysis runs
    - Semantic versioning (major.minor.patch)
    - Provenance tracking with git integration
    - Result comparison tools
    - Metadata preservation
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the archive manager.
        
        Args:
            base_dir: Base directory for the project (defaults to current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.archive_dir = self.base_dir / "archives"
        self.current_dir = self.base_dir / "current"
        self.config_file = self.base_dir / "archive_config.json"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize directories
        self._initialize_directories()
        
        # Load configuration
        self.config = self._load_config()
    
    def _initialize_directories(self):
        """Create necessary directory structure."""
        self.archive_dir.mkdir(exist_ok=True)
        self.current_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized archival
        for subdir in ["analysis_scripts", "results", "figures", "tables", "logs"]:
            (self.archive_dir / subdir).mkdir(exist_ok=True)
            (self.current_dir / subdir).mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load archive configuration or create default."""
        default_config = {
            "version": "1.0.0",
            "last_archive": None,
            "archive_retention_days": 365,
            "auto_archive_enabled": True,
            "files_to_track": [
                "*.py",
                "*.R",
                "*.ipynb",
                "results/*.csv",
                "results/*.json",
                "results/*.png",
                "results/*.pdf",
                "tables/*.csv",
                "tables/*.tex"
            ],
            "exclude_patterns": [
                "__pycache__",
                ".pytest_cache",
                "*.pyc",
                ".DS_Store"
            ]
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _increment_version(self, version_type: str = "minor") -> str:
        """
        Increment version number.
        
        Args:
            version_type: Type of version increment ('major', 'minor', 'patch')
            
        Returns:
            New version string
        """
        current_version = self.config["version"]
        major, minor, patch = map(int, current_version.split('.'))
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid version_type: {version_type}")
        
        return f"{major}.{minor}.{patch}"
    
    def _get_git_info(self) -> Dict:
        """Get current git repository information."""
        try:
            repo = git.Repo(self.base_dir)
            return {
                "commit_hash": repo.head.commit.hexsha,
                "commit_message": repo.head.commit.message.strip(),
                "branch": repo.active_branch.name,
                "is_dirty": repo.is_dirty(),
                "untracked_files": repo.untracked_files
            }
        except (git.InvalidGitRepositoryError, git.exc.GitError):
            self.logger.warning("Not a git repository or git error occurred")
            return {}
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_manifest(self, archive_path: Path, files: List[Path]) -> Dict:
        """
        Create manifest with file metadata.
        
        Args:
            archive_path: Path to archive directory
            files: List of files being archived
            
        Returns:
            Manifest dictionary
        """
        manifest = {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": self.config["version"],
            "git_info": self._get_git_info(),
            "files": {},
            "archive_path": str(archive_path),
            "total_files": len(files)
        }
        
        for file_path in files:
            if file_path.exists():
                relative_path = file_path.relative_to(self.base_dir)
                manifest["files"][str(relative_path)] = {
                    "size_bytes": file_path.stat().st_size,
                    "modified_time": datetime.datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                    "hash": self._calculate_file_hash(file_path)
                }
        
        return manifest
    
    def archive_current_analysis(self, version_type: str = "minor", 
                               description: str = None) -> str:
        """
        Archive current analysis to timestamped directory.
        
        Args:
            version_type: Version increment type ('major', 'minor', 'patch')
            description: Optional description of this archive
            
        Returns:
            Path to created archive
        """
        # Increment version
        new_version = self._increment_version(version_type)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"v{new_version}_{timestamp}"
        
        if description:
            # Sanitize description for directory name
            safe_desc = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_desc = safe_desc.replace(' ', '_')[:50]  # Limit length
            archive_name += f"_{safe_desc}"
        
        archive_path = self.archive_dir / archive_name
        archive_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Creating archive: {archive_name}")
        
        # Find files to archive
        files_to_archive = []
        for pattern in self.config["files_to_track"]:
            files_to_archive.extend(self.base_dir.glob(pattern))
        
        # Remove excluded patterns
        for exclude_pattern in self.config["exclude_patterns"]:
            files_to_archive = [
                f for f in files_to_archive 
                if not any(part.startswith(exclude_pattern.rstrip('*')) 
                          for part in f.parts)
            ]
        
        # Copy files to archive
        for file_path in files_to_archive:
            if file_path.is_file():
                relative_path = file_path.relative_to(self.base_dir)
                dest_path = archive_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest_path)
        
        # Create manifest
        manifest = self._create_manifest(archive_path, files_to_archive)
        if description:
            manifest["description"] = description
        
        manifest_path = archive_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        # Clear figures folder after archiving (keeping only current analysis)
        self._clear_archived_figures()
        
        # Update configuration
        self.config["version"] = new_version
        self.config["last_archive"] = {
            "timestamp": timestamp,
            "version": new_version,
            "path": str(archive_path),
            "description": description
        }
        self._save_config(self.config)
        
        self.logger.info(f"Archive created successfully: {archive_path}")
        return str(archive_path)
    
    def list_archives(self) -> List[Dict]:
        """
        List all available archives with metadata.
        
        Returns:
            List of archive information dictionaries
        """
        archives = []
        
        for archive_dir in self.archive_dir.iterdir():
            if archive_dir.is_dir():
                manifest_path = archive_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    archives.append({
                        "name": archive_dir.name,
                        "path": str(archive_dir),
                        "version": manifest.get("version", "unknown"),
                        "timestamp": manifest.get("timestamp", "unknown"),
                        "description": manifest.get("description", ""),
                        "total_files": manifest.get("total_files", 0)
                    })
        
        # Sort by timestamp (newest first)
        archives.sort(key=lambda x: x["timestamp"], reverse=True)
        return archives
    
    def restore_from_archive(self, archive_name: str, 
                           target_dir: str = None) -> bool:
        """
        Restore files from specified archive.
        
        Args:
            archive_name: Name of archive to restore from
            target_dir: Target directory (defaults to current directory)
            
        Returns:
            True if successful, False otherwise
        """
        archive_path = self.archive_dir / archive_name
        if not archive_path.exists():
            self.logger.error(f"Archive not found: {archive_name}")
            return False
        
        target_path = Path(target_dir) if target_dir else self.base_dir
        
        # Load manifest
        manifest_path = archive_path / "manifest.json"
        if not manifest_path.exists():
            self.logger.error(f"Manifest not found in archive: {archive_name}")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.logger.info(f"Restoring from archive: {archive_name}")
        
        # Copy files from archive
        for relative_path in manifest["files"]:
            source_path = archive_path / relative_path
            dest_path = target_path / relative_path
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
            else:
                self.logger.warning(f"File missing in archive: {relative_path}")
        
        self.logger.info("Restore completed successfully")
        return True
    
    def compare_archives(self, archive1: str, archive2: str) -> Dict:
        """
        Compare two archives and return differences.
        
        Args:
            archive1: First archive name
            archive2: Second archive name
            
        Returns:
            Dictionary containing comparison results
        """
        def load_manifest(archive_name):
            manifest_path = self.archive_dir / archive_name / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            return None
        
        manifest1 = load_manifest(archive1)
        manifest2 = load_manifest(archive2)
        
        if not manifest1 or not manifest2:
            return {"error": "Could not load one or both manifests"}
        
        files1 = set(manifest1["files"].keys())
        files2 = set(manifest2["files"].keys())
        
        comparison = {
            "archive1": {"name": archive1, "version": manifest1.get("version")},
            "archive2": {"name": archive2, "version": manifest2.get("version")},
            "files_only_in_1": list(files1 - files2),
            "files_only_in_2": list(files2 - files1),
            "common_files": list(files1 & files2),
            "modified_files": []
        }
        
        # Check for modifications in common files
        for file_path in comparison["common_files"]:
            hash1 = manifest1["files"][file_path]["hash"]
            hash2 = manifest2["files"][file_path]["hash"]
            if hash1 != hash2:
                comparison["modified_files"].append({
                    "file": file_path,
                    "size_change": (
                        manifest2["files"][file_path]["size_bytes"] - 
                        manifest1["files"][file_path]["size_bytes"]
                    )
                })
        
        return comparison
    
    def cleanup_old_archives(self, retention_days: int = None) -> List[str]:
        """
        Remove archives older than specified retention period.
        
        Args:
            retention_days: Number of days to retain archives (uses config if None)
            
        Returns:
            List of removed archive names
        """
        if retention_days is None:
            retention_days = self.config["archive_retention_days"]
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        removed_archives = []
        
        for archive_dir in self.archive_dir.iterdir():
            if archive_dir.is_dir():
                manifest_path = archive_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    archive_date = datetime.datetime.fromisoformat(
                        manifest.get("timestamp", "1970-01-01T00:00:00")
                    )
                    
                    if archive_date < cutoff_date:
                        shutil.rmtree(archive_dir)
                        removed_archives.append(archive_dir.name)
                        self.logger.info(f"Removed old archive: {archive_dir.name}")
        
        return removed_archives
    
    def _clear_archived_figures(self):
        """
        Clear figures folder after archiving, keeping only recent files.
        This ensures only current analysis images remain in figures/.
        """
        figures_dir = self.base_dir / "figures"
        if not figures_dir.exists():
            return
        
        # Define cutoff time (keep files newer than 1 hour)
        import time
        cutoff_time = time.time() - 3600  # 1 hour ago
        
        removed_files = []
        figure_extensions = ['.png', '.pdf', '.svg', '.jpg', '.jpeg']
        
        for file_path in figures_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in figure_extensions:
                # Check file modification time
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path.name))
                        self.logger.info(f"Removed archived figure: {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove figure {file_path.name}: {e}")
        
        if removed_files:
            self.logger.info(f"Cleared {len(removed_files)} archived figures from figures/ directory")
        else:
            self.logger.info("No old figures to clear from figures/ directory")

def main():
    """Command line interface for archive manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HEAT Analysis Archive Manager")
    parser.add_argument("command", choices=["archive", "list", "restore", "compare", "cleanup"])
    parser.add_argument("--version-type", choices=["major", "minor", "patch"], default="minor")
    parser.add_argument("--description", help="Archive description")
    parser.add_argument("--archive1", help="First archive for comparison")
    parser.add_argument("--archive2", help="Second archive for comparison")
    parser.add_argument("--target", help="Target archive name for restore")
    parser.add_argument("--retention-days", type=int, help="Retention period for cleanup")
    
    args = parser.parse_args()
    
    manager = AnalysisArchiveManager()
    
    if args.command == "archive":
        archive_path = manager.archive_current_analysis(
            version_type=args.version_type,
            description=args.description
        )
        print(f"Archive created: {archive_path}")
    
    elif args.command == "list":
        archives = manager.list_archives()
        print(f"Found {len(archives)} archives:")
        for archive in archives:
            print(f"  {archive['name']} (v{archive['version']}) - {archive['description']}")
    
    elif args.command == "restore":
        if not args.target:
            print("Error: --target required for restore command")
            return
        success = manager.restore_from_archive(args.target)
        print(f"Restore {'successful' if success else 'failed'}")
    
    elif args.command == "compare":
        if not args.archive1 or not args.archive2:
            print("Error: --archive1 and --archive2 required for compare command")
            return
        comparison = manager.compare_archives(args.archive1, args.archive2)
        print(json.dumps(comparison, indent=2))
    
    elif args.command == "cleanup":
        removed = manager.cleanup_old_archives(args.retention_days)
        print(f"Removed {len(removed)} old archives")

if __name__ == "__main__":
    main()