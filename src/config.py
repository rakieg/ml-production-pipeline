"""Configuration management."""
import os
import subprocess
import json


class Config:
    
    def __init__(self, environment='dev'):
        """Initialize configuration."""
        
        if environment not in ['dev', 'staging', 'prod']:
            raise ValueError(f"Invalid environment: {environment}. Must be dev, staging, or prod")

        self.environment = environment
        self.host = os.getenv('DATABRICKS_HOST')
        self.token = os.getenv('DATABRICKS_TOKEN')
        self.user_email = os.getenv('DATABRICKS_USER_EMAIL', 'ml-pipeline')
        
        if not self.host or not self.token:
            raise ValueError("Set DATABRICKS_HOST and DATABRICKS_TOKEN")
        
        # Set environment variables for CLI
        os.environ['DATABRICKS_HOST'] = self.host
        os.environ['DATABRICKS_TOKEN'] = self.token
        
        # Paths
        self.workspace_base = f"/Workspace/Users/{self.user_email}/ml-pipeline"
        self.workspace_folder = f"{self.workspace_base}/{environment}"
        
        # Names
        self.model_name = f"workspace.{environment}.{environment}_breast_cancer_classifier"
        self.training_job = f"{environment}-training"
        self.inference_job = f"{environment}-inference"
        
        # Detect CLI command
        self.cli_cmd = self._get_cli_command()
        
        print(f"Environment: {environment}")
        print(f"User: {self.user_email}")
        print(f"Workspace: {self.workspace_folder}")
    
    def _get_cli_command(self):
        """Get databricks CLI command."""
        try:
            subprocess.run(['databricks', '--version'], 
                         capture_output=True, check=True, timeout=5)
            return ['databricks']
        except:
            return ['python', '-m', 'databricks.cli']
