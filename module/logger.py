import os
import time
import h5py
import mlflow
from mlflow.entities import Metric
import numpy as np


class MLFlowLogger:
    """Simple MLFlow logger for tracking experiments"""

    def __init__(self, run_id, mlflow_host="127.0.0.1", mlflow_port=5000):
        self.run_id = run_id
        self.mlflow_host = mlflow_host
        self.mlflow_port = mlflow_port

        # Set tracking URI
        mlflow.set_tracking_uri(f"http://{mlflow_host}:{mlflow_port}")

        # Get base path for artifacts
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        # Handle different artifact URI schemes
        if artifact_uri.startswith("file://"):
            self.base_path = artifact_uri.replace("file://", "")
        elif artifact_uri.startswith("mlflow-artifacts:"):
            # For mlflow-artifacts, use the default artifact location
            self.base_path = os.path.join("mlruns", run.info.experiment_id, run_id, "artifacts")
        else:
            # Fallback: remove any scheme prefix
            self.base_path = artifact_uri.split("://")[-1] if "://" in artifact_uri else artifact_uri

    def log_param(self, key, value):
        """Log a parameter to MLflow"""
        if value is not None:
            mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        """Log a metric to MLflow"""
        mlflow.log_metric(key, value, step=step)

    def log_hdf5(self, data, data_split, data_type, step):
        """Log HDF5 data as artifact"""
        # Format step
        if step is not None:
            step_str = f"_{step:0>7}"
        else:
            step_str = ""

        # Create artifact directory
        artifact_dir = os.path.join(self.base_path, data_split, data_type)
        os.makedirs(artifact_dir, exist_ok=True)

        # Create file path
        file_path = os.path.join(artifact_dir, f"{data_type}{step_str}.h5")

        # Save data to HDF5
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=data)

    def log_text(self, text, data_split, data_type, step):
        """Log text data as artifact"""
        # Create artifact directory
        artifact_dir = os.path.join(self.base_path, data_split)
        os.makedirs(artifact_dir, exist_ok=True)

        # Create file path
        file_path = os.path.join(artifact_dir, f"{data_type}_step{step}.txt")

        # Save text to file
        if isinstance(text, (list, tuple)):
            text = "\n".join(str(t) for t in text)

        with open(file_path, "w") as f:
            f.write(str(text))

    def log_samples_metrics(self, data, prefix, step):
        """Log individual sample metrics"""
        if isinstance(data, (np.ndarray, list)):
            data_array = np.array(data) if isinstance(data, list) else data

            # Log each sample individually
            client = mlflow.MlflowClient()
            for i in range(len(data_array)):
                metric = Metric(
                    key=f"sample_{i:0>3}_{prefix}",
                    value=float(data_array[i]),
                    timestamp=int(time.time() * 1000),
                    step=step,
                )
                client.log_metric(self.run_id, metric.key, metric.value, metric.timestamp, metric.step)


