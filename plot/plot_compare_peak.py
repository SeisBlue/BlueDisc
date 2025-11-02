import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import mlflow


def get_mlflow_client():
    """Get MLflow client"""
    with open("/workspace/hosts.json", "r") as f:
        hosts = json.load(f)

    mlflow_host = hosts.get("mlflow_host", "0.0.0.0")
    mlflow_port = hosts.get("mlflow_port", 5000)
    return mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")


def get_experiment_name_from_run(run_id):
    """Get the corresponding experiment name from run ID"""
    try:
        client = get_mlflow_client()
        run = client.get_run(run_id)
        experiment = client.get_experiment(run.info.experiment_id)
        return experiment.name
    except Exception as e:
        print(f"Error getting experiment name for run {run_id}: {e}")
        return None


def load_peak_data(run_id, data_split, max_step=None):
    """Load time error and height data for all steps of the specified run"""
    client = get_mlflow_client()
    experiment_id = client.get_run(run_id).info.experiment_id
    base_path = f"/workspace/mlartifacts/{experiment_id}/{run_id}/artifacts"
    matching_results_dir = os.path.join(base_path, data_split, "matching_results")

    if not os.path.exists(matching_results_dir):
        print(f"Error: Matching results directory not found: {matching_results_dir}")
        return None

    csv_files = sorted(
        glob(os.path.join(matching_results_dir, "matching_results_*.csv"))
    )
    if not csv_files:
        print(f"Error: No matching results CSV files found in {matching_results_dir}")
        return None

    print(f"Found {len(csv_files)} matching results files for run {run_id}")

    # Filter files that meet the max_step condition
    if max_step is not None:
        valid_files = []
        for csv_file in csv_files:
            step = int(
                os.path.basename(csv_file)
                .replace("matching_results_", "")
                .replace(".csv", "")
            )
            if step < max_step:
                valid_files.append(csv_file)
        csv_files = valid_files

    if not csv_files:
        print(f"No valid files found for run {run_id} with max_step {max_step}")
        return None

    # Merge data from all steps
    all_data_rows = []
    processed_steps = []
    peaks_summary = []  # Added: to store peak count information for each trace

    for csv_file in csv_files:
        step = int(
            os.path.basename(csv_file)
            .replace("matching_results_", "")
            .replace(".csv", "")
        )

        try:
            df = pd.read_csv(csv_file)

            # Expand position_errors and heights lists
            for _, row in df.iterrows():
                # Store peak count information
                peaks_summary.append(
                    {
                        "phase": row["phase"],
                        "trace_name": row["trace_name"],
                        "step": step,
                        "num_pred_peaks": row.get("num_pred_peaks", 0),
                        "num_label_peaks": row.get("num_label_peaks", 0),
                    }
                )

                position_errors = (
                    eval(row["position_errors"])
                    if isinstance(row["position_errors"], str)
                    else row["position_errors"]
                )
                heights = (
                    eval(row["heights"])
                    if isinstance(row["heights"], str)
                    else row["heights"]
                )

                # Ensure position_errors and heights are lists
                if not isinstance(position_errors, list):
                    position_errors = []
                if not isinstance(heights, list):
                    heights = []

                for pos_error, height in zip(position_errors, heights):
                    all_data_rows.append(
                        {
                            "position_error": pos_error,  # Keep sign
                            "height": height,
                            "phase": row["phase"],
                            "trace_name": row["trace_name"],
                            "step": step,
                        }
                    )

            processed_steps.append(step)

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    if not all_data_rows:
        print(f"No valid data found for run {run_id}")
        return None

    result_df = pd.DataFrame(all_data_rows)
    peaks_summary_df = pd.DataFrame(peaks_summary)

    # Attach peaks summary to result_df for later use
    result_df.peaks_summary = peaks_summary_df

    print(
        f"Loaded {len(result_df)} data points from {len(processed_steps)} steps for run {run_id}"
    )
    print(f"Steps included: {sorted(processed_steps)}")
    return result_df


def plot_combined_jointplot(
    run1_data,
    run2_data,
    run3_data,
    run1_label,
    run2_label,
    run3_label,
    save_dir,
    data_split,
):
    """Combine six run + phase combinations into one large image (3x2 layout)"""

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create large figure - 3x2 layout, each subplot is a complete jointplot
    fig = plt.figure(figsize=(30, 15))

    # Create subplots for each run + phase combination
    runs_data = [
        (run1_data, run1_label, "run1"),
        (run2_data, run2_label, "run2"),
        (run3_data, run3_label, "run3"),
    ]
    phases = ["P", "S"]

    # Create main 3x2 layout - run1-P, run1-S, run2-P, run2-S, run3-P, run3-S
    subplot_positions = [
        (0, 0),
        (0, 1),  # run1: P, S
        (1, 0),
        (1, 1),  # run2: P, S
        (2, 0),
        (2, 1),  # run3: P, S
    ]

    # Unit conversion constant: 100Hz sampling rate, 1 second = 100 samples
    SAMPLING_RATE = 100  # Hz

    time_range = 3.5

    plot_idx = 0
    for run_data, run_label, run_type in runs_data:
        for phase in phases:
            if run_data is None:
                plot_idx += 1
                continue

            row, col = subplot_positions[plot_idx]
            plot_idx += 1

            phase_data = run_data[run_data["phase"] == phase]
            if len(phase_data) == 0:
                continue

            # Create GridSpec for each subplot
            gs_sub = fig.add_gridspec(
                2,
                2,
                left=0.05 + col * 0.45,
                right=0.05 + col * 0.45 + 0.4,
                bottom=0.02 + (2 - row) * 0.31,
                top=0.02 + (2 - row) * 0.31 + 0.27,
                height_ratios=[0.5, 1],
                width_ratios=[2, 0.5],
                hspace=0.02,
                wspace=0.01,
            )

            # Create axes for the subplot
            ax_joint = fig.add_subplot(gs_sub[1, 0])
            ax_top = fig.add_subplot(gs_sub[0, 0], sharex=ax_joint)
            ax_right = fig.add_subplot(gs_sub[1, 1], sharey=ax_joint)
            ax_text = fig.add_subplot(gs_sub[0, 1])  # Upper right blank area for text

            # Prepare data - convert position_error from sample count to seconds
            x = phase_data["position_error"] / SAMPLING_RATE  # Convert to seconds
            y = phase_data["height"]

            # Set color mapping - assign different colors for three runs
            if run_type == "run1":
                if phase == "P":
                    cmap = "Greens"
                    edge_color = "#2E8B57"
                else:
                    cmap = "Purples"
                    edge_color = "#663399"
            elif run_type == "run2":
                if phase == "P":
                    cmap = "Blues"
                    edge_color = "#1E90FF"
                else:
                    cmap = "Oranges"
                    edge_color = "#FF8C00"
            else:  # run3
                if phase == "P":
                    cmap = "BuGn"  # Cool colors: blue-green
                    edge_color = "#008B8B"  # Dark cyan
                else:
                    cmap = "Reds"  # Warm colors: red
                    edge_color = "#DC143C"  # Crimson

            # Plot 2D histogram - range converted from [-250, 250] samples to [-2.5, 2.5] seconds
            h = ax_joint.hist2d(
                x,
                y,
                bins=[100, 25],
                range=[[-time_range, time_range], [0, 1]],  # Range in seconds
                cmap=cmap,
                alpha=0.8,
                norm=plt.matplotlib.colors.LogNorm(vmin=1),
            )

            # Add Gaussian curve - height 1, position at 0, variance 0.2s
            x_gaussian = np.linspace(-2.5, 2.5, 1000)
            sigma = 0.2  # Standard deviation 0.2s
            mu = 0  # Position at 0
            height_scale = 1  # Height 1
            gaussian_curve = height_scale * np.exp(
                -0.5 * ((x_gaussian - mu) / sigma) ** 2
            )

            # Plot Gaussian curve, using thicker black line
            ax_joint.plot(
                x_gaussian,
                gaussian_curve,
                "k-",
                linewidth=1,
                alpha=0.3,
                label="Gaussian (μ=0, σ=0.2s)",
            )

            # Add colorbar in the bottom left corner of the figure, horizontal placement
            cbar_ax = ax_joint.inset_axes(
                [0.68, 0.35, 0.3, 0.02]
            )  # [x, y, width, height]
            cbar = plt.colorbar(h[3], cax=cbar_ax, orientation="horizontal")
            cbar.set_label("Count", fontsize=14)
            cbar.ax.tick_params(labelsize=14)

            # Add standard lines - ensure displayed on top layer
            # Position standard lines (±0.1 seconds)
            ax_joint.axvline(
                x=0.1,
                color="red",
                linestyle="-",
                linewidth=1,
                alpha=0.3,
                zorder=10,
                label="±0.1s standard",
            )
            ax_joint.axvline(
                x=-0.1, color="red", linestyle="-", linewidth=1, alpha=0.3, zorder=10
            )
            # Height standard line (0.7)
            ax_joint.axhline(
                y=0.7,
                color="red",
                linestyle="-",
                linewidth=1,
                alpha=0.3,
                zorder=10,
                label="0.7 height standard",
            )

            # Plot upper 1D histogram
            ax_top.hist(
                x,
                bins=100,
                range=[-time_range, time_range],
                color=edge_color,
                alpha=0.3,
                edgecolor="white",
                linewidth=0.5,
                label="All data",
            )

            # Data with height ≥ 0.7 (dark foreground)
            x_high_height = x[y >= 0.7]
            if len(x_high_height) > 0:
                ax_top.hist(
                    x_high_height,
                    bins=100,
                    range=[-time_range, time_range],
                    color=edge_color,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                    label="Height ≥ 0.7",
                )

            # Also add position error standard lines to upper histogram
            ax_top.axvline(x=0.1, color="red", linestyle="-", linewidth=1, alpha=0.3)
            ax_top.axvline(x=-0.1, color="red", linestyle="-", linewidth=1, alpha=0.3)
            ax_top.set_ylabel("Count", fontsize=14)
            ax_top.tick_params(axis="x", labelbottom=False, labelsize=14)
            ax_top.tick_params(axis="y", labelsize=14)

            # # Modify y-axis tick labels to display in units of 1000
            # y_ticks = ax_top.get_yticks()
            # ax_top.set_yticklabels([f"{int(tick/1000)}" for tick in y_ticks])

            ax_top.grid(True, alpha=0.3)
            ax_top.legend(fontsize=14, loc="upper right")

            # Plot right-side 1D histogram
            ax_right.hist(
                y,
                bins=25,
                range=[0, 1],
                orientation="horizontal",
                color=edge_color,
                alpha=0.3,
                edgecolor="white",
                linewidth=0.5,
                label="All data",
            )

            # Data with position error within ±0.1s (dark foreground)
            y_within_range = y[(x >= -0.1) & (x <= 0.1)]
            if len(y_within_range) > 0:
                ax_right.hist(
                    y_within_range,
                    bins=25,
                    range=[0, 1],
                    orientation="horizontal",
                    color=edge_color,
                    alpha=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                    label="Within ±0.1s",
                )

            # Also add height standard line to right histogram
            ax_right.axhline(y=0.7, color="red", linestyle="-", linewidth=1, alpha=0.3)

            ax_right.set_xlabel("Count", fontsize=14)
            ax_right.tick_params(axis="x", labelsize=14)
            ax_right.tick_params(axis="y", labelleft=False, labelsize=14)

            # # Modify x-axis tick labels to display in units of 1000
            # x_ticks = ax_right.get_xticks()
            # ax_right.set_xticklabels([f"{int(tick/1000)}" for tick in x_ticks])

            ax_right.grid(True, alpha=0.3)
            ax_right.legend(fontsize=12, loc="lower right")

            # Set axis labels and title for main plot
            ax_joint.set_xlabel("Position Error (seconds)", fontsize=14)
            ax_joint.set_ylabel("Height", fontsize=14)
            ax_joint.tick_params(labelsize=14)
            ax_joint.grid(True, alpha=0.3)
            ax_joint.set_xlim(-time_range, time_range)
            ax_joint.set_ylim(0, 1.18)

            # Add statistical information
            n_points = len(phase_data)
            mean_error = x.mean()
            std_error = x.std()

            # Get total peak count for this phase from peaks summary
            phase_peaks_summary = run_data.peaks_summary[
                run_data.peaks_summary["phase"] == phase
            ]
            total_pred_peaks = phase_peaks_summary["num_pred_peaks"].sum()
            total_label_peaks = phase_peaks_summary["num_label_peaks"].sum()

            # Calculate ratios and counts
            matched_ratio = (
                (n_points / total_pred_peaks * 100) if total_label_peaks > 0 else 0
            )

            within_pos_std = ((x >= -0.1) & (x <= 0.1)).sum()
            pos_std_ratio = (
                within_pos_std / total_label_peaks * 100 if total_label_peaks > 0 else 0
            )

            high_height = (y >= 0.7).sum()
            height_std_ratio = (
                high_height / total_label_peaks * 100 if total_label_peaks > 0 else 0
            )

            both_std = ((x >= -0.1) & (x <= 0.1) & (y >= 0.7)).sum()
            both_std_ratio = (
                both_std / total_label_peaks * 100 if total_label_peaks > 0 else 0
            )

            pos_good_height_bad = ((x >= -0.1) & (x <= 0.1) & (y < 0.7)).sum()
            pos_good_height_bad_ratio = (
                pos_good_height_bad / total_label_peaks * 100
                if total_label_peaks > 0
                else 0
            )

            # Add text description in upper right blank area - keep only first line
            ax_text.axis("off")
            text_str = f"{run_label}\n" f"{phase} Phase"
            ax_text.text(
                0,
                0.5,
                text_str,
                va="center",
                ha="left",
                fontsize=20,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor="white",
                    alpha=0,
                ),
            )

            # Add statistical information in bottom right corner of middle 2D histogram
            stats_text = f"Label Count: {total_label_peaks} (100%)\n"
            stats_text += f"Over & Precise (TP): {both_std} ({both_std_ratio:.1f}%)\n"
            stats_text += f"Below & Precise: {pos_good_height_bad} ({pos_good_height_bad_ratio:.1f}%)\n"

            ax_joint.text(
                0.02,
                0.4,
                stats_text,
                transform=ax_joint.transAxes,
                va="top",
                ha="left",
                fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    edgecolor="none",
                    facecolor="white",
                    alpha=0,
                ),
            )

    # Save combined image
    filename = f"compare_peak.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Combined joint plot (3 runs) saved: {save_path}")


def plot_peak_jointplot(
    run1_data,
    run2_data,
    run3_data,
    run1_label,
    run2_label,
    run3_label,
    save_dir,
    data_split,
):
    """Create combined joint plot"""
    plot_combined_jointplot(
        run1_data,
        run2_data,
        run3_data,
        run1_label,
        run2_label,
        run3_label,
        save_dir,
        data_split,
    )


def print_data_summary(data, label):
    """Print data summary"""
    # Unit conversion constant: 100Hz sampling rate, 1 second = 100 samples
    SAMPLING_RATE = 100  # Hz

    print(f"\n=== {label} Data Summary ===")
    for phase in ["P", "S"]:
        phase_data = data[data["phase"] == phase]
        if len(phase_data) > 0:
            # Convert Time_error from samples to seconds
            position_seconds = phase_data["position_error"] / SAMPLING_RATE
            print(f"{phase} Phase:")
            print(f"  Count: {len(phase_data)}")
            print(
                f"  Position Error - Mean: {position_seconds.mean():.3f}s, "
                f"Std: {position_seconds.std():.3f}s, "
                f"Max: {position_seconds.max():.3f}s"
            )
            print(
                f"  Height - Mean: {phase_data['height'].mean():.4f}, "
                f"Std: {phase_data['height'].std():.4f}, "
                f"Max: {phase_data['height'].max():.4f}"
            )
        else:
            print(f"{phase} Phase: No data")


def get_save_path(args, run_id):
    """Get save path"""
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        return args.save_dir

    client = get_mlflow_client()
    experiment_id = client.get_run(run_id).info.experiment_id
    base_path = f"/workspace/mlartifacts/{experiment_id}/{run_id}/artifacts"
    return os.path.join(base_path, args.data_split, "matching_results")


def main():
    parser = argparse.ArgumentParser(
        description="Plot position error vs height jointplot from evaluator CSV files"
    )
    parser.add_argument(
        "--run-id", type=str, required=True, help="MLflow run ID (first run)"
    )
    parser.add_argument(
        "--compare-run-id", type=str, help="MLflow run ID to compare (second run)"
    )
    parser.add_argument(
        "--compare-run-id-2", type=str, help="MLflow run ID to compare (third run)"
    )
    parser.add_argument("--max-step", type=int, help="Maximum step to use")
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["track", "train", "dev", "test"],
        help="Data split to analyze",
    )
    parser.add_argument("--save-dir", type=str, help="Directory to save the plot")
    args = parser.parse_args()

    # Load data for first run
    print(f"Loading first run: {args.run_id}")
    run1_data = load_peak_data(args.run_id, args.data_split, args.max_step)
    if run1_data is None:
        print("Failed to load first run data")
        return

    # Get experiment name
    run1_label = get_experiment_name_from_run(args.run_id) or f"Run {args.run_id[:8]}"
    print(f"First run label: {run1_label}")

    # Load data for second run
    run2_data = None
    run2_label = "No Comparison"
    if args.compare_run_id:
        print(f"Loading second run: {args.compare_run_id}")
        run2_data = load_peak_data(args.compare_run_id, args.data_split, args.max_step)
        if run2_data is None:
            print("Failed to load second run data, proceeding without second run")
        else:
            run2_label = (
                get_experiment_name_from_run(args.compare_run_id)
                or f"Run {args.compare_run_id[:8]}"
            )
            print(f"Second run label: {run2_label}")

    # Load data for third run
    run3_data = None
    run3_label = "No Comparison"
    if args.compare_run_id_2:
        print(f"Loading third run: {args.compare_run_id_2}")
        run3_data = load_peak_data(
            args.compare_run_id_2, args.data_split, args.max_step
        )
        if run3_data is None:
            print("Failed to load third run data, proceeding without third run")
        else:
            run3_label = (
                get_experiment_name_from_run(args.compare_run_id_2)
                or f"Run {args.compare_run_id_2[:8]}"
            )
            print(f"Third run label: {run3_label}")

    # Set save path
    save_dir = get_save_path(args, args.run_id)

    plot_peak_jointplot(
        run1_data,
        run2_data,
        run3_data,
        run1_label,
        run2_label,
        run3_label,
        save_dir,
        args.data_split,
    )

    # Print data summary
    print_data_summary(run1_data, run1_label)
    if run2_data is not None:
        print_data_summary(run2_data, run2_label)
    if run3_data is not None:
        print_data_summary(run3_data, run3_label)


if __name__ == "__main__":
    main()
