import argparse
import json
import os
import shutil
import sys
import glob

import h5py
import mlflow

from module.evaluator import get_picks, match_peaks_and_calculate_errors
from module.parallel import parallel

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, required=True)
parser.add_argument("--max-step", type=int)
parser.add_argument(
    "--data-split",
    type=str,
    default="test",
    choices=["track", "train", "dev", "test"],
)

args = parser.parse_args()

with open("/workspace/hosts.json", "r") as f:
    hosts = json.load(f)

mlflow_host = hosts.get("mlflow_host", "0.0.0.0")
mlflow_port = hosts.get("mlflow_port", 5000)

run_id = args.run_id
client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")
experiment_id = client.get_run(run_id).info.experiment_id
experiment_name = client.get_experiment(experiment_id).name
base_path = f"/workspace/mlartifacts/{experiment_id}/{run_id}/artifacts"

data_split = args.data_split
os.makedirs(os.path.join(base_path, data_split), exist_ok=True)

if args.max_step is not None:
    max_step = args.max_step
else:
    prediction_dir = os.path.join(base_path, data_split, "prediction")
    max_step = len(glob.glob(os.path.join(prediction_dir, "prediction_*.h5")))

match_confidence = 0.1
match_tolerance = 3000
precise_confidence = 0.7
precise_tolerance = 10
min_peak_distance = 100

# remove existing matching results directory if it exists
matching_results_path = os.path.join(base_path, data_split, "matching_results")
if os.path.exists(matching_results_path):
    shutil.rmtree(matching_results_path)
os.makedirs(matching_results_path, exist_ok=True)


def process_step(step):
    if data_split == "track":
        label_path = os.path.join(base_path, data_split, "label", f"label_0000000.h5")
        trace_names_path = os.path.join(
            base_path, data_split, "trace_name", f"trace_name_0000000.txt"
        )
    else:
        label_path = os.path.join(
            base_path, data_split, "label", f"label_{step:0>7}.h5"
        )
        trace_names_path = os.path.join(
            base_path, data_split, "trace_name", f"trace_name_{step:0>7}.txt"
        )

    with h5py.File(label_path, "r") as f:
        labels = f["data"][:]

    pred_path = os.path.join(
        base_path, data_split, "prediction", f"prediction_{step:0>7}.h5"
    )
    with h5py.File(pred_path, "r") as f:
        predictions = f["data"][:]

    with open(trace_names_path, "r") as f:
        trace_names = f.readlines()

    trace_names = [name.strip() for name in trace_names]

    pred_picks = get_picks(
        predictions, confidence=match_confidence, distance=min_peak_distance
    )
    label_picks = get_picks(
        labels, confidence=match_confidence, distance=min_peak_distance
    )

    # 配對peaks並計算誤差
    matching_results_dataframe = match_peaks_and_calculate_errors(
        pred_picks,
        label_picks,
        trace_names,
        tolerance=match_tolerance,
        precision_confidence=precise_confidence,
        precision_tolerance=precise_tolerance,
    )
    matching_results_dataframe.to_csv(
        os.path.join(
            base_path,
            data_split,
            "matching_results",
            f"matching_results_{step:0>7}.csv",
        ),
        index=False,
    )
    print(f"matching_results_{step:0>7}.csv")
    sys.stdout.flush()


steps = range(max_step)
parallel(steps, process_step, cpu_count=os.cpu_count())
