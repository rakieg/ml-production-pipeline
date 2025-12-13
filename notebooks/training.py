# Databricks notebook training source

# Environment parameter will be overwritten from job parameter.
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

# Construct model name based on environment
model_name = f"workspace.{env}.{env}_breast_cancer_classifier"
print(f"Environment: {env}")
print(f"Model: {model_name}")

# COMMAND ----------

# Setup Unity Catalog schema for environment
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS workspace.{env}")
    print(f"Unity Catalog schema ready: workspace.{env}")
except Exception as e:
    print(f"Schema setup: {e}")

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import tempfile
import os

# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Feature engineering
X['mean_radius_squared'] = X['mean radius'] ** 2
X['mean_texture_log'] = np.log1p(X['mean texture'])
X['area_perimeter_ratio'] = X['mean area'] / (X['mean perimeter'] + 1e-6)
X['radius_texture_interaction'] = X['mean radius'] * X['mean texture']

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")

# COMMAND ----------

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# COMMAND ----------

# Configure MLflow
current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
experiment_name = f"/Users/{current_user}/ml-pipeline/{env}/breast_cancer_classification"
mlflow.set_experiment(experiment_name)

print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# Train model
with mlflow.start_run(run_name=f"training_{env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    
    # Hyperparameters
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    random_state = 42
    
    # Log parameters
    mlflow.log_param("environment", env)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("n_features", X_train_scaled.shape[1])
    
    # Train
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    print("\nModel Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    
    # Log model
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train_scaled, y_pred)
    
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name=model_name
    )
    
    # Log scaler
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "preprocessor")
    
    run_id = run.info.run_id
    print(f"\nRun ID: {run_id}")
    print(f"Model: {model_name}")

# COMMAND ----------

# Verify registration
from mlflow.tracking import MlflowClient
client = MlflowClient()

try:
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if model_versions:
        latest = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
        print(f"Model registered as version {latest.version}")
        print(f"URI: models:/{model_name}/{latest.version}")
except Exception as e:
    print(f"Check MLflow UI for registration status: {e}")
