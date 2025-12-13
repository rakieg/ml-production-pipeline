"""Workspace management."""
import subprocess
import os


class WorkspaceManager:
    """Manages workspace folders and notebooks."""
    
    def __init__(self, config):
        """Initialize workspace manager."""
        self.config = config
    
    def setup(self):
        """Setup workspace."""
        self.create_folders()
        self.upload_notebooks()
    
    def create_folders(self):
        """Create workspace folders."""
        print("\n Creating folders...")
        
        for folder in [self.config.workspace_base, self.config.workspace_folder]:
            try:
                result = subprocess.run(
                    self.config.cli_cmd + ['workspace', 'mkdirs', folder],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    print(f"  {folder}")
                else:
                    # Check if it already exists
                    check = subprocess.run(
                        self.config.cli_cmd + ['workspace', 'ls', folder],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if check.returncode == 0:
                        print(f"  {folder} (already exists)")
                    else:
                        print(f"  Failed to create {folder}")
                        print(f"  Error: {result.stderr}")
                        raise Exception(f"Could not create folder: {folder}")
                        
            except subprocess.TimeoutExpired:
                print(f"  Timeout creating {folder}")
            except Exception as e:
                print(f"  Error: {e}")
                raise
    
    def upload_notebooks(self):
        """Upload notebooks."""
        print("\nUploading notebooks...")
        
        notebooks = [
            ('notebooks/training.py', f"{self.config.workspace_folder}/training"),
            ('notebooks/inference.py', f"{self.config.workspace_folder}/inference")
        ]
        
        for local_path, remote_path in notebooks:
            if not os.path.exists(local_path):
                print(f"  {local_path} not found - skipping")
                continue
                
            try:
                result = subprocess.run(
                    self.config.cli_cmd + [
                        'workspace', 'import',
                        local_path, remote_path,
                        '--language', 'PYTHON',
                        '--format', 'SOURCE',
                        '--overwrite'
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"  {remote_path}")
                else:
                    print(f"  Failed to upload {remote_path}")
                    print(f"  Error: {result.stderr}")
                    # Don't raise, continue with other notebook
                    
            except subprocess.TimeoutExpired:
                print(f"  Timeout uploading {local_path}")
            except Exception as e:
                print(f"  Error: {e}")
