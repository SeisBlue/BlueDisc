import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist


def get_precision(tp, fp):
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def get_recall(tp, fn):
    if tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


def get_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def get_picks(batch_data, confidence=0.7, distance=100):
    batch_picks = []
    for batch in batch_data:
        p_peaks, p_properties = find_peaks(
            batch[0], distance=distance, height=confidence
        )
        s_peaks, s_properties = find_peaks(
            batch[1], distance=distance, height=confidence
        )
        batch_picks.append(
            {
                "P": {"peaks": p_peaks, "heights": p_properties["peak_heights"]},
                "S": {"peaks": s_peaks, "heights": s_properties["peak_heights"]},
            }
        )

    return batch_picks


def match_peaks_and_calculate_errors(
    pred_picks,
    label_picks,
    trace_names,
    tolerance=500,
    precision_tolerance=10,
    precision_confidence=0.7,
):
    """
    Match predicted peaks with labeled peaks and calculate errors.

    Args:
        pred_picks: List of predicted peaks
        label_picks: List of labeled peaks
        tolerance: Maximum allowed distance for matching
        precision_tolerance: Distance threshold for a "precise" match (default 10,
            roughly 0.1s at a 100 Hz sampling rate)
        precision_confidence: Minimum peak height required for a precise match

    Returns:
        matched_results: A pandas DataFrame containing matching results and errors
    """
    batch_results = []
    for batch_idx, (pred_batch, label_batch, trace_name) in enumerate(
        zip(pred_picks, label_picks, trace_names)
    ):
        for phase in ["P", "S"]:
            pred_peaks = pred_batch[phase]["peaks"]
            label_peaks = label_batch[phase]["peaks"]
            pred_heights = pred_batch[phase]["heights"]
            label_heights = label_batch[phase]["heights"]

            # If either set of peaks is empty, handle the boundary case
            if len(pred_peaks) == 0 or len(label_peaks) == 0:
                phase_result = {
                    "trace_name": trace_name,
                    "phase": phase,
                    "batch_idx": batch_idx,
                    "matched_pairs": [],
                    "unmatched_pred": list(range(len(pred_peaks))),
                    "unmatched_label": list(range(len(label_peaks))),
                    "position_errors": [],
                    "heights": [],
                    "precise_matches": [],
                    "total_distance": 0,
                    "num_matches": 0,
                    "num_precise_matches": 0,
                    "num_pred_peaks": len(pred_peaks),
                    "num_label_peaks": len(label_peaks),
                }
                batch_results.append(phase_result)
                continue

            # Compute the distance matrix between all predicted and labeled peaks
            pred_positions = pred_peaks.reshape(-1, 1)
            label_positions = label_peaks.reshape(-1, 1)
            distance_matrix = cdist(pred_positions, label_positions, metric="euclidean")

            # Use a greedy algorithm to match peaks (smallest distance first)
            matched_pairs = []
            used_pred = set()
            used_label = set()
            position_errors = []
            heights = []
            precise_matches = []

            # Create a list of (distance, pred_index, label_index) and sort it
            distance_pairs = []
            for i in range(len(pred_peaks)):
                for j in range(len(label_peaks)):
                    distance_pairs.append((distance_matrix[i, j], i, j))

            distance_pairs.sort(key=lambda x: x[0])

            # Greedy matching
            for distance, pred_idx, label_idx in distance_pairs:
                if (
                    distance <= tolerance
                    and pred_idx not in used_pred
                    and label_idx not in used_label
                ):
                    matched_pairs.append((pred_idx, label_idx))
                    used_pred.add(pred_idx)
                    used_label.add(label_idx)

                    # Calculate position error and record peak height
                    pos_error = int(pred_peaks[pred_idx] - label_peaks[label_idx])
                    height = float(pred_heights[pred_idx])

                    position_errors.append(pos_error)
                    heights.append(height)

                    # Determine whether this is a precise match
                    is_precise = (
                        distance <= precision_tolerance
                        and height >= precision_confidence
                    )
                    precise_matches.append(is_precise)

            # Find unmatched peaks
            unmatched_pred = [i for i in range(len(pred_peaks)) if i not in used_pred]
            unmatched_label = [
                i for i in range(len(label_peaks)) if i not in used_label
            ]

            total_distance = sum(
                distance_matrix[pred_idx, label_idx]
                for pred_idx, label_idx in matched_pairs
            )

            phase_result = {
                "trace_name": trace_name,
                "phase": phase,
                "batch_idx": batch_idx,
                "matched_pairs": matched_pairs,
                "unmatched_pred": unmatched_pred,
                "unmatched_label": unmatched_label,
                "position_errors": position_errors,
                "heights": heights,
                "precise_matches": precise_matches,
                "total_distance": total_distance,
                "num_matches": len(matched_pairs),
                "num_precise_matches": sum(precise_matches),
                "num_pred_peaks": len(pred_peaks),
                "num_label_peaks": len(label_peaks),
            }

            batch_results.append(phase_result)

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(batch_results)

    return df


def compute_precision_recall_f1(self, batch_confusion_matrix):
    batch_metrics = {
        "P": {"tp": 0, "fp": 0, "fn": 0},
        "S": {"tp": 0, "fp": 0, "fn": 0},
    }
    for sample_id, metrics in batch_confusion_matrix:
        for phase in "PS":
            for key in batch_metrics[phase].keys():
                batch_metrics[phase][key] += metrics[phase][key]

    for phase in "PS":
        tp = batch_metrics[phase]["tp"]
        fp = batch_metrics[phase]["fp"]
        fn = batch_metrics[phase]["fn"]
        precision = self.get_precision(tp, fp)
        recall = self.get_recall(tp, fn)
        f1 = self.get_f1_score(precision, recall)

        batch_metrics[phase]["precision"] = precision
        batch_metrics[phase]["recall"] = recall
        batch_metrics[phase]["f1"] = f1

    return batch_metrics


def compute_tp_fp_fn(
    self, pred_data, label_data, confidence=0.7, threshold=0.1, sampling_rate=100
):
    threshold = threshold * sampling_rate
    metrics = {"P": {}, "S": {}}
    for i, phase in enumerate("PS"):
        label_peaks = self.get_peaks(label_data[i], confidence=confidence)
        pred_peaks = self.get_peaks(pred_data[i], confidence=confidence)

        tp = 0
        fp = 0
        fn = 0

        if len(label_peaks) == 0:
            metrics[phase] = {
                "tp": 0,
                "fp": len(pred_peaks),
                "fn": 0,
            }
            continue

        if len(pred_peaks) == 0:
            metrics[phase] = {
                "tp": 0,
                "fp": 0,
                "fn": len(label_peaks),
            }
            continue

        for label_peak in label_peaks:
            if np.min(np.abs(label_peak - pred_peaks)) <= threshold:
                tp += 1
            else:
                fn += 1

        for pred_peak in pred_peaks:
            if np.min(np.abs(pred_peak - label_peaks)) > threshold:
                fp += 1

        metrics[phase] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return metrics


class AsyncEvaluator:
    def __init__(self, run_id, mlflow_host="0.0.0.0", mlflow_port=8080):
        self.run_id = run_id
        self.client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")
        self.experiment_id = self.client.get_run(run_id).info.experiment_id
        self.base_path = (
            f"/workspace/mlartifacts/{self.experiment_id}/{self.run_id}/artifacts"
        )

        self.manager = multiprocessing.Manager()
        self.eval_queue = self.manager.Queue()
        self.processing_count = self.manager.Value("i", 0)

        self.p = [
            multiprocessing.Process(target=self.eval_sample_daemon),
        ]

    def start_daemon(self):
        for process in self.p:
            process.start()

    def loading_animation(self):
        remain_queue = self.eval_queue.qsize()
        loading_chars = ["-", "\\", "|", "/"]

        # Rotate through loading animation characters
        for char in loading_chars:
            # Clear the previous character
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()

            # Display the current loading character
            sys.stdout.write(
                f"Waiting for logging sample prediction, remaining:{remain_queue} {char}"
            )
            sys.stdout.flush()
            time.sleep(0.1)

    def stop_daemon(self):
        # Wait for all sample logging to complete
        while True:
            if self.eval_queue.empty() and self.processing_count.value == 0:
                break
            self.loading_animation()

        for process in self.p:
            process.terminate()
            process.join()

    def put_eval_samples(self, pred_data, label_data, step):
        self.eval_queue.put((pred_data, label_data, step))

    def eval_sample_daemon(self):
        while True:
            try:
                if self.eval_queue.empty():
                    time.sleep(0.1)
                    continue

                pred_data, label_data, step = self.eval_queue.get(timeout=60)

                def process_sample(sample_id):
                    metrics = compute_tp_fp_fn(
                        pred_data[sample_id],
                        label_data[sample_id],
                    )
                    return sample_id, metrics

                with ThreadPoolExecutor(max_workers=100) as executor:
                    batch_confusion_matrix = sorted(
                        list(executor.map(process_sample, range(len(pred_data)))),
                        key=lambda x: x[0],
                    )

                batch_metrics = compute_precision_recall_f1(batch_confusion_matrix)
                self.log_metric("P_precision", batch_metrics["P"]["precision"], step)
                self.log_metric("P_recall", batch_metrics["P"]["recall"], step)
                self.log_metric("P_f1", batch_metrics["P"]["f1"], step)
                self.log_metric("S_precision", batch_metrics["S"]["precision"], step)
                self.log_metric("S_recall", batch_metrics["S"]["recall"], step)
                self.log_metric("S_f1", batch_metrics["S"]["f1"], step)

                self.eval_queue.task_done()

            except self.eval_queue.Empty:
                break

            except Exception as e:
                print(e.__class__.__name__, e)
                print("Terminating eval_sample_daemon")
                break

    def log_metric(self, key, value, step):
        self.client.log_metric(
            run_id=self.run_id,
            key=key,
            value=value,
            timestamp=int(time.time() * 1000),
            step=step,
        )
