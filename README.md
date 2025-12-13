# MLOps Production Pipeline

Production-ready ML pipeline for Databricks with automated deployment, model training, and inference.

## Architecture

- **Training Job**: Scheduled monthly, trains Random Forest classifier with MLflow tracking
- **Inference Job**: Scheduled daily, loads registered model and runs predictions
- **CI/CD**: Automated deployment via CLI tool
- **Environment Separation**: Dev, staging and prod environments with isolated resources

## Instructions to run Locally

1.Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```
2.Configure environment variables in `.env` for local runs:   
``` 
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com     
DATABRICKS_TOKEN=your-token     
DATABRICKS_USER_EMAIL=your-email@example.com    
```

3.Setup Workspace & Deploy Jobs/pipelines:
```bash 
python -m src.cli --env <ENV> --setup
```

### Commands
```bash
# Deploy to dev (default)
python -m src.cli

# Staging Workspace setup & Deploy jobs
python -m src.cli --env staging --setup

# Deploy to prod along with workspace setup
python -m src.cli --env prod --setup

# Update jobs only (no workspace setup)
python -m src.cli --env staging

# Run diagnostics
python -m src.cli --test
```

## Design

### Object-Oriented Structure
- **Config**: Environment configuration and CLI detection
- **WorkspaceManager**: Folder creation and notebook uploads
- **JobManager**: Job creation and updates via Databricks CLI

### Environment Separation
- Three environments: dev, staging, prod
- Separate workspace folders per environment
- Environment-specific model naming (e.g., `workspace.prod.prod_breast_cancer_classifier`)
- Isolated MLflow experiments per environment

### Deployment Strategy
- Idempotent operations (safe to run multiple times)
- Automatic detection of existing resources
- Update-in-place for jobs

## Environment Details

| Environment | Use Case | Schedule |
|-------------|----------|----------|
| **dev** | Development and testing | On-demand |
| **staging** | Pre-production validation | Monthly training, Daily inference |
| **prod** | Production deployment | Monthly training, Daily inference |

### Workspace Structure
/Workspace/Users/{user}/ml-pipeline/    
├── dev/    
│   ├── training (notebook)     
│   └── inference (notebook)    
├── staging/    
│   ├── training (notebook)     
│   └── inference (notebook)    
└── prod/   
├── training (notebook)     
└── inference (notebook)        

### Unity Catalog Structure
workspace (catalog)     
├── dev (schema)    
│   └── dev_breast_cancer_classifier (model)    
├── staging (schema)    
│   └── staging_breast_cancer_classifier (model)    
└── prod (schema)   
    └── prod_breast_cancer_classifier (model)   

### Model Registry
- Dev: `workspace.dev.dev_breast_cancer_classifier`
- Staging: `workspace.staging.staging_breast_cancer_classifier`
- Prod: `workspace.prod.prod_breast_cancer_classifier`

## Requirements Met

- Python CLI tool with object-oriented design
- Two Databricks jobs per environment with schedules (monthly/daily)
- MLflow tracking and model registry
- GitHub Actions integration ready
- Environment separation (dev/staging/prod)
