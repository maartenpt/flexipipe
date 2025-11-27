"""
TEITOK-style backup functionality for writeback operations.

Creates once-a-day backups to a backups folder in the execution directory.
For a file like xmlfiles/a/b/something.xml, backs up to backups/something-YYYYMMDD.xml
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Optional


def create_teitok_backup(file_path: str, execution_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create a TEITOK-style backup of a file if one doesn't exist for today.
    
    For a file like `xmlfiles/a/b/something.xml`, creates a backup at
    `backups/something-YYYYMMDD.xml` in the execution directory.
    
    Args:
        file_path: Path to the file to backup
        execution_dir: Directory where backups folder should be created (defaults to current working directory)
        
    Returns:
        Path to the backup file if created, None if backup already exists for today
    """
    source_path = Path(file_path)
    if not source_path.exists():
        return None
    
    # Use execution directory or current working directory
    if execution_dir is None:
        execution_dir = Path.cwd()
    else:
        execution_dir = Path(execution_dir)
    
    # Create backups directory if it doesn't exist
    backups_dir = execution_dir / "backups"
    backups_dir.mkdir(exist_ok=True)
    
    # Get filename without path (e.g., "something.xml" from "xmlfiles/a/b/something.xml")
    filename = source_path.name
    
    # Create backup filename: filename-YYYYMMDD.xml
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    
    # Split filename into name and extension
    if "." in filename:
        name_part, ext_part = filename.rsplit(".", 1)
        backup_filename = f"{name_part}-{date_str}.{ext_part}"
    else:
        backup_filename = f"{filename}-{date_str}"
    
    backup_path = backups_dir / backup_filename
    
    # Only create backup if it doesn't already exist for today
    if backup_path.exists():
        return None
    
    # Copy file to backup location
    try:
        shutil.copy2(source_path, backup_path)
        return backup_path
    except Exception:
        # If backup fails, return None (don't fail the main operation)
        return None


def should_create_backup(file_path: str, execution_dir: Optional[Path] = None) -> bool:
    """
    Check if a backup should be created (i.e., doesn't exist for today).
    
    Args:
        file_path: Path to the file to check
        execution_dir: Directory where backups folder is located
        
    Returns:
        True if backup should be created, False if backup already exists for today
    """
    source_path = Path(file_path)
    if not source_path.exists():
        return False
    
    if execution_dir is None:
        execution_dir = Path.cwd()
    else:
        execution_dir = Path(execution_dir)
    
    backups_dir = execution_dir / "backups"
    if not backups_dir.exists():
        return True
    
    # Check if backup for today exists
    filename = source_path.name
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    
    if "." in filename:
        name_part, ext_part = filename.rsplit(".", 1)
        backup_filename = f"{name_part}-{date_str}.{ext_part}"
    else:
        backup_filename = f"{filename}-{date_str}"
    
    backup_path = backups_dir / backup_filename
    return not backup_path.exists()

