# BlueDisc: Adversarial Shape Learning for Seismic Phase Picking

This repo is a minimal, reproducible implementation to validate the paper “Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning.” It augments a PhaseNet generator with a lightweight conditional discriminator (BlueDisc) to enforce shape-then-align learning, which eliminates the 0.5-amplitude suppression band and increases effective S-phase detections.

- Paper: see the accompanying paper for full details
- Core idea: combine BCE (temporal anchoring) with a cGAN shape critic to decouple shape learning from temporal alignment
- Scope: this code is for verification only; for details, please refer to the paper

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
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns
```

Train
- BCE only (no GAN):
```bash
python 01_training.py \
  --label D \
  --dataset Instance \
  --batch-size 100 \
  --max-steps 10000
```
- Shape-then-align (cGAN): set a data loss weight (λ), e.g. 4000 per paper
```bash
python 01_training.py \
  --label D \
  --dataset Instance \
  --data-weight 4000 \
  --g-lr 1e-3 --d-lr 1e-3 \
  --batch-size 100 \
  --max-steps 10000
```
Notes
- `--dataset` is a SeisBench dataset class name (e.g., `Instance`, `ETHZ`). The dataset will be downloaded by SeisBench on first use.
- `--label` controls the output channel order: `D` (detection) or `N` (noise) after `PS`.

Infer
1) Find the `run_id` from MLflow UI or `mlruns/*/*/meta.yaml`.
2) Run inference (choose split and optional checkpoint by step/epoch):
```bash
python 02_inference.py \
  --run-id <RUN_ID> \
  --dataset Instance \
  --data-split dev
```

Evaluate
```bash
python 03_evaluation.py \
  --run-id <RUN_ID> \
  --data-split dev
```
Outputs are saved under `mlruns/<experiment>/<run_id>/artifacts/` (waveforms, labels, predictions as HDF5; checkpoints under `checkpoint/`; matching CSVs under `<split>/matching_results/`).

## Repo layout
- `01_training.py`, `02_inference.py`, `03_evaluation.py`: train → infer → evaluate
- `module/`: generator (PhaseNet wrapper), discriminator (BlueDisc), GAN training loop, data pipeline, logger
- `mlruns/`: MLflow experiments and artifacts
- `docs/`: short documentation

For CLI details, tips, and caveats, see `docs/README.md`. Please cite the paper when using this code.
