# BlueDisc docs (short)

This repository reproduces the paper’s key finding: a cGAN-based shape-then-align strategy fixes amplitude suppression in seismic S-phase picking.

- Core modules: `module/generator.py` (PhaseNet wrapper), `module/discriminator.py` (BlueDisc), `module/gan_model.py` (training loop)
- Analysis scripts: `loss_landscape_analysis.py`, `plot_compare_*.py` for figure-style comparisons

## Reproduce the main result (short recipe)

1) Start MLflow
```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```
2) Train (hybrid, λ=4000, 10k steps)
```bash
python 01_training.py --label D --dataset Instance \
  --data-weight 4000 --batch-size 100 --max-steps 10000
```
3) Get the `run_id` from MLflow UI or `mlruns/*/*/meta.yaml`
4) Inference (dev split)
```bash
python 02_inference.py --run-id <RUN_ID> --dataset Instance --data-split dev
```
5) Evaluate (dev split)
```bash
python 03_evaluation.py --run-id <RUN_ID> --data-split dev
```
Check outputs under `mlruns/<exp>/<run>/artifacts/dev/` and matching CSVs in `matching_results/`.

## Usage summary

Training
- BCE only (no GAN): omit `--data-weight`
- Hybrid (cGAN): set `--data-weight` to λ (paper uses 4000)
- Device: `--device` accepts `auto` (default), `cuda`, `mps` (Apple Silicon), or `cpu`

Inference and evaluation
- `02_inference.py` writes waveforms/labels/predictions to `mlruns/.../artifacts/<split>/`
- `03_evaluation.py` matches peaks, writes CSVs to `<split>/matching_results/`

## CLI options (key)
- `01_training.py`
  - `--label {D,N}`: choose detection or noise as the third output channel (after `PS`)
  - `--dataset <SeisBenchClass>`: e.g., `Instance`, `ETHZ`; auto-downloaded by SeisBench
  - `--data-weight <float>`: λ for BCE in the hybrid loss (enables GAN when provided)
  - `--g-lr`, `--d-lr`: learning rates (default 1e-3)
  - `--batch-size` (default 100), `--max-steps` (default 10000), `--device` (`auto`)
- `02_inference.py`
  - `--run-id <id>`: MLflow run id to load checkpoints and store outputs
  - `--dataset <SeisBenchClass>`; `--data-split {track,train,dev,test}` (default `test`)
  - `--step` or `--epoch` to select a specific checkpoint; otherwise latest
- `03_evaluation.py`
  - `--run-id <id>`; `--data-split` as above; `--max-step` to cap processed batches

## Data pipeline
- SeisBench generators with augmentations in `module/pipeline.py`
- Probabilistic labels (Gaussian, σ=20) for P/S/Noise via SeisBench
- Detection channel `D` uses `TaperedDetectionLabeller` (Gaussian-tapered window from P to extended S), see `module/labeler.py`

## Tips
- Start MLflow locally before running scripts; the code expects `http://127.0.0.1:5000` in training/inference. Evaluation script defaults to `http://0.0.0.0:5000`; align your MLflow server host or edit the constant in `03_evaluation.py` if needed.
- On Apple Silicon, set `--device mps` to use the Metal backend; on NVIDIA, `--device cuda`.
- If PyTorch is missing, install it per your platform; requirements.txt does not pin torch.

## Troubleshooting
- “CUDA not available”: use `--device cpu` or install CUDA-enabled PyTorch.
- “Dataset class not found”: ensure the name matches SeisBench (e.g., `Instance`, `ETHZ`).
- MLflow connection errors: make sure the server runs on the host/port used in scripts (127.0.0.1 or 0.0.0.0 on port 5000). Use `--backend-store-uri ./mlruns --default-artifact-root ./mlruns` so artifacts land under this repo.

For deeper background, results, and design rationale, please refer to the accompanying paper.
