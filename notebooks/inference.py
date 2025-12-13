# Databricks notebook source

# Get environment parameter first, env will be overwritten in Job
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")

# Construct model name based on environment
model_name = f"workspace.{env}.{env}_breast_cancer_classifier"
print(f"Environment: {env}")
print(f"Model: {model_name}")

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# Load model
from mlflow.tracking import MlflowClient
import time

client = MlflowClient()

model_uri = None
model_stage = None

# Wait for model to be ready (handle PENDING_REGISTRATION status)
def wait_for_model_ready(model_name, timeout=300):
    """Wait for model version to be ready."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No model versions found yet, waiting...")
                time.sleep(5)
                continue
            
            latest = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            
            if latest.status == "READY":
                return latest
            else:
                print(f"Model version {latest.version} status: {latest.status}, waiting...")
                time.sleep(5)
        except Exception as e:
            print(f"Checking model status: {e}")
            time.sleep(5)
    
    raise Exception(f"Model not ready after {timeout} seconds")

try:
    # Try to load champion alias
    model_uri = f"models:/{model_name}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    model_stage = "Champion"
    print("Loaded champion model")
except:
    # Fall back to latest version, wait if needed
    try:
        print("Champion alias not found, loading latest version...")
        latest_version = wait_for_model_ready(model_name)
        
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        model_stage = f"Version {latest_version.version}"
        print(f"Loaded version {latest_version.version}")
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

print(f"Model URI: {model_uri}")

# COMMAND ----------

# Load sample data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

np.random.seed(int(datetime.now().timestamp()) % 1000)
sample_indices = np.random.choice(X.index, size=50, replace=False)
X_inference = X.loc[sample_indices].copy()

# Apply feature engineering
X_inference['mean_radius_squared'] = X_inference['mean radius'] ** 2
X_inference['mean_texture_log'] = np.log1p(X_inference['mean texture'])
X_inference['area_perimeter_ratio'] = X_inference['mean area'] / (X_inference['mean perimeter'] + 1e-6)
X_inference['radius_texture_interaction'] = X_inference['mean radius'] * X_inference['mean texture']

print(f"Inference dataset: {len(X_inference)} records")

# COMMAND ----------

# Load scaler
try:
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if model_versions:
        latest = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0]
        run_id = latest.run_id
        
        scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/scaler.pkl")
        scaler = joblib.load(scaler_path)
        print("Loaded scaler from artifacts")
except:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_inference)
    print("Warning: Created new scaler (not recommended for production)")

X_inference_scaled = scaler.transform(X_inference)

# COMMAND ----------

# Run predictions
predictions = model.predict(X_inference_scaled)
prediction_probabilities = model.predict_proba(X_inference_scaled)

results = pd.DataFrame({
    'record_id': sample_indices,
    'prediction': predictions,
    'prediction_label': ['Malignant' if p == 0 else 'Benign' for p in predictions],
    'probability_malignant': prediction_probabilities[:, 0],
    'probability_benign': prediction_probabilities[:, 1],
    'inference_timestamp': datetime.now(),
    'environment': env,
    'model_stage': model_stage
})

results['high_confidence'] = (
    (results['probability_benign'] > 0.8) | 
    (results['probability_malignant'] > 0.8)
)

print("\nInference Results:")
print(f"  Total predictions: {len(results)}")
print(f"  Malignant: {(predictions == 0).sum()}")
print(f"  Benign: {(predictions == 1).sum()}")
print(f"  High confidence: {results['high_confidence'].sum()}")

display(results.head(10))

# COMMAND ----------

# Save results
output_table = f"{env}_inference_results"
output_path = f"/ml-pipeline/{env}/inference_results"

spark_df = spark.createDataFrame(results)

spark_df.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .saveAsTable(output_table)

print(f"Results saved to Delta table: {output_table}")

# COMMAND ----------

# Log to MLflow
current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
experiment_name = f"/Users/{current_user}/ml-pipeline/{env}/breast_cancer_inference"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"inference_{env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_param("environment", env)
    mlflow.log_param("model_uri", model_uri)
    mlflow.log_param("model_stage", model_stage)
    mlflow.log_param("n_records", len(results))
    
    mlflow.log_metric("predicted_malignant", (predictions == 0).sum())
    mlflow.log_metric("predicted_benign", (predictions == 1).sum())
    mlflow.log_metric("high_confidence_ratio", results['high_confidence'].mean())
    
    print("Inference metrics logged to MLflow")
