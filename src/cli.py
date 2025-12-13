"""Command-line interface for MLOps pipeline deployment."""
import click
import sys
import subprocess
from dotenv import load_dotenv
from src.config import Config
from src.workspace import WorkspaceManager
from src.jobs import JobManager

load_dotenv()


@click.command()
@click.option('--env', '-e', default='dev', type=click.Choice(['dev', 'staging', 'prod']))
@click.option('--setup', is_flag=True, help='Setup workspace folders and notebooks')
@click.option('--test', is_flag=True, help='Run diagnostics')
def main(env, setup, test):
    """
    Deploy ML pipeline to Databricks.

    By default, deploys jobs only.
    Use --setup to also create folders and upload notebooks.
    Use --test to run diagnostics.
    """

    try:
        print("="*70)
        print("MLOps Pipeline Deployment")
        print("="*70)

        config = Config(env)

        if test:
            _run_diagnostics(config)
            return

        if setup:
            workspace = WorkspaceManager(config)
            workspace.setup()

        jobs = JobManager(config)
        training_id, inference_id = jobs.deploy_all()

        print("\n" + "="*70)
        print("Deployment Complete")
        print("="*70)
        print(f"\nJob URLs:")
        print(f"  Training:  {config.host}/#job/{training_id}")
        print(f"  Inference: {config.host}/#job/{inference_id}")
        print(f"\nNext Steps:")
        print(f"  1. Schemas are auto-created per environment:")
        print(f"     workspace.dev, workspace.staging, workspace.prod")
        print(f"  2. Run training job manually to create initial model")
        print(f"  3. Run inference job to test predictions")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _run_diagnostics(config):
    """Run diagnostic tests."""
    print("\nRunning diagnostics...\n")

    print("1. Testing CLI command...")
    try:
        result = subprocess.run(
            config.cli_cmd + ['--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"   CLI command: {' '.join(config.cli_cmd)}")
        print(f"   Version: {result.stdout.strip()}")
    except Exception as e:
        print(f"   CLI test failed: {e}")

    print("\n2. Testing workspace access...")
    try:
        result = subprocess.run(
            config.cli_cmd + ['workspace', 'ls', '/Workspace/Users'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   Workspace access: OK")
        else:
            print(f"   Workspace access failed: {result.stderr}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Testing folder creation...")
    test_folder = f"{config.workspace_base}/test"
    try:
        result = subprocess.run(
            config.cli_cmd + ['workspace', 'mkdirs', test_folder],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"   Created test folder: {test_folder}")
        else:
            print(f"   Failed to create folder: {result.stderr}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()