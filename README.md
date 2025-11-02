# BlueDisc: Adversarial Shape Learning for Seismic Phase Picking

This repo is a minimal, reproducible implementation to validate the paper “Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning.” It augments a PhaseNet generator with a lightweight conditional discriminator (BlueDisc) to enforce label shape learning, which eliminates the 0.5-amplitude suppression band and increases effective S-phase detections.

- Core idea: combine BCE Loss with a cGAN shape critic to decouple shape learning from temporal alignment

<img src="docs/fig/model_architecture.png" alt="BlueDisc architecture" width="400" />

## Quick start

Prereqs
- Python 3.10+
- PyTorch (install per your platform: https://pytorch.org/get-started/locally/)
- MLflow 2.x (already in requirements)

Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Install PyTorch separately per platform (CPU/CUDA/MPS), e.g.:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Start MLflow (required)
```bash
mlflow ui
# or
python -m mlflow ui
```

Train
- BCE only (no GAN):
```bash
python 01_training.py \
  --label N \
  --dataset InstanceCounts \
  --max-steps 10000
```
- Conditional GAN: set a data loss weight (λ), e.g. 4000 per paper
```bash
python 01_training.py \
  --label N \
  --dataset InstanceCounts \
  --data-weight 4000 \
  --max-steps 10000
```
Notes
- `--dataset` is a [SeisBench dataset class name](https://seisbench.readthedocs.io/en/stable/pages/documentation/data.html#seisbench.data.instance.InstanceCounts) (e.g., `InstanceCounts`, `ETHZ`). The dataset will be downloaded by SeisBench on first use.
- `--label` controls the output channel order: `N` (noise) or  `D` (detection).

Infer
1) Find the `run_id` from MLflow UI or `mlruns/*/*/meta.yaml`.
2) Run inference (choose split and optional checkpoint by step/epoch):
```bash
python 02_inference.py \
  --run-id <RUN_ID> \
  --dataset InstanceCounts 
```

Evaluate
```bash
python 03_evaluation.py \
  --run-id <RUN_ID> 
```
Outputs are saved under `mlruns/<experiment>/<run_id>/artifacts/` (waveforms, labels, predictions as HDF5; checkpoints under `checkpoint/`; matching CSVs under `<split>/matching_results/`).

## Visualization

The repository includes several plotting scripts to analyze model behavior during and after training:

### Training-based visualization (using logged tracking data)
During training, the model automatically logs sample predictions at each step. You can visualize training progression using:
- `plot_compare_runs.py`: side-by-side comparison of predictions from different runs at the same step

<img src="docs/fig/compare_runs.png" alt="compare runs" width="400" />

- `plot_compare_shape.py`: compare prediction shapes at selected training steps
- `plot_compare_time.py`: visualize how predictions evolve over training steps for a specific sample

<img src="docs/fig/training_dynamics.png" alt="training dynamics" width="400" />


These scripts work directly with the tracking data logged during training (`mlruns/<experiment>/<run_id>/artifacts/track/`).

### Inference-based visualization (requires test dataset predictions)
- `plot_compare_peak.py`: analyze peak detection accuracy by comparing predicted peaks with ground-truth labels. **Requires running both inference (`02_inference.py`) and evaluation (`03_evaluation.py`)** on the test dataset first. The evaluation step generates matching results (`matching_results/` CSVs) that pair each predicted peak with its corresponding label peak, enabling quantitative analysis of detection performance.

<img src="docs/fig/s_distribution.png" alt="s_distribution" width="400" />

### Data exploration
- `plot_compare_phase.py`: visualize P and S phase label arrangements in the dataset. This is a data exploration tool independent of model training.

<img src="docs/fig/compare_p_s.png" alt="compare_p_s" width="400" />

## Repo layout
- `01_training.py`, `02_inference.py`, `03_evaluation.py`: train → infer → evaluate pipeline
- `module/`: generator (PhaseNet wrapper), discriminator (BlueDisc), GAN training loop, data pipeline, logger
- `plot_*.py`: visualization scripts for analyzing training, inference, and data
- `mlruns/`: MLflow experiments and artifacts
- `docs/`: short documentation
- `loss_landscape/`: standalone loss-landscape simulations (BCE toy experiments)
  - `loss_landscape_analysis.py`: BCE loss surface visualization (height vs. time/peak)
  - `no_model_bce_test.py`: point-wise vs Gaussian-parameterized BCE optimization

For CLI details, tips, and caveats, see `docs/README.md`. Please cite the paper when using this code.
