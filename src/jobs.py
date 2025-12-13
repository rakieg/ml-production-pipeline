"""Job management."""
import subprocess
import json
import tempfile
import os


class JobManager:
    """Manages Databricks jobs."""
    
    def __init__(self, config):
        """Initialize job manager."""
        self.config = config

    def deploy_all(self):
        """Deploy both jobs."""
        print("\n Deploying jobs...")

        training_id = self.deploy_job(
            self.config.training_job,
            self._training_config()
        )
        print(f"  Training: {training_id}")
        
        inference_id = self.deploy_job(
            self.config.inference_job,
            self._inference_config()
        )
        print(f"  Inference: {inference_id}")
        
        return training_id, inference_id
    
    def deploy_job(self, name, config):
        """Deploy single job."""
        existing_id = self._find_job(name)
        
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            config_file = f.name
        
        try:
            if existing_id:
                # Update existing
                result = subprocess.run(
                    self.config.cli_cmd + ['jobs', 'reset', '--job-id', existing_id, '--json-file', config_file],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
                
                if result.returncode == 0:
                    return existing_id
                else:
                    print(f"   Update failed: {result.stderr}")
                    raise Exception(f"Failed to update job: {result.stderr}")
            else:
                # Create new
                result = subprocess.run(
                    self.config.cli_cmd + ['jobs', 'create', '--json-file', config_file],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    return str(data['job_id'])
                else:
                    print(f"  Create failed: {result.stderr}")
                    raise Exception(f"Failed to create job: {result.stderr}")
        finally:
            os.unlink(config_file)
    
    def _find_job(self, name):
        """Find job by name."""
        try:
            result = subprocess.run(
                self.config.cli_cmd + ['jobs', 'list', '--output', 'json'],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            data = json.loads(result.stdout)
            
            for job in data.get('jobs', []):
                if job.get('settings', {}).get('name') == name:
                    return str(job['job_id'])
        except:
            pass
        return None
    
    def _training_config(self):
        """Training job config."""
        return {
            "name": self.config.training_job,
            "tasks": [{
                "task_key": "train",
                "notebook_task": {
                    "notebook_path": f"{self.config.workspace_folder}/training",
                    "base_parameters": {
                        "env": self.config.environment,
                        "model_name": self.config.model_name
                    }
                }
            }],
            "schedule": {
                "quartz_cron_expression": "0 0 0 1 * ?",
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1
        }
    
    def _inference_config(self):
        """Inference job config."""
        return {
            "name": self.config.inference_job,
            "tasks": [{
                "task_key": "infer",
                "notebook_task": {
                    "notebook_path": f"{self.config.workspace_folder}/inference",
                    "base_parameters": {
                        "env": self.config.environment,
                        "model_name": self.config.model_name
                    }
                }
            }],
            "schedule": {
                "quartz_cron_expression": "0 0 0 * * ?",
                "timezone_id": "UTC",
                "pause_status": "UNPAUSED"
            },
            "max_concurrent_runs": 1
        }
