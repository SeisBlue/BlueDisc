import argparse
import os

import mlflow
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.gan_model import GANModel
from module.logger import MLFlowLogger
from module.pipeline import AugmentationsBuilder
from module.random_seed import RandomSeedManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["track", "train", "dev", "test"],
    )

    args = parser.parse_args()

    seed_value = 42
    seed_manager = RandomSeedManager(seed_value)
    seed_manager.set_seed()

    # MLflow settings - using local setup
    mlflow_host = "127.0.0.1"
    mlflow_port = 5000

    run_id = args.run_id
    client = mlflow.MlflowClient(f"http://{mlflow_host}:{mlflow_port}")
    experiment_id = client.get_run(run_id).info.experiment_id
    experiment_name = client.get_experiment(experiment_id).name

    # Initialize logger
    logger = MLFlowLogger(
        run_id=run_id,
        log_training=False,
        mlflow_host=mlflow_host,
        mlflow_port=mlflow_port
    )

    base_path = logger.base_path
    checkpoint_dir = os.path.join(base_path, "checkpoint")

    gan_model = GANModel()
    gan_model.load_checkpoint_by_dir(checkpoint_dir, step=args.step, epoch=args.epoch)
    g_model = gan_model.g_model

    # Load dataset
    data = sbd.InstanceCounts(sampling_rate=100)
    train, dev, test = data.train_dev_test()

    aug_builder = AugmentationsBuilder(dataset=data)
    augmentations = aug_builder.build()

    test_generator = sbg.GenericGenerator(dev)
    test_generator.add_augmentations(augmentations)

    g = torch.Generator()
    g.manual_seed(seed_value)

    num_workers = os.cpu_count() or 1  # Handle None case

    test_loader = DataLoader(
        test_generator,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_manager.worker_init_fn,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )


    size = len(test_generator)  # Use generator length instead of loader.dataset
    batch_size = test_loader.batch_size
    gan_model.eval()
    with torch.no_grad():
        with tqdm(total=size, desc="Inferencing", ncols=80) as pbar:
            for batch_id, batch in enumerate(test_loader):
                if batch_id == size:
                    break
                trace_name_list = batch.pop("trace_name", "")
                logger.log_text(
                    trace_name_list,
                    data_split=args.data_split,
                    data_type="trace_name",
                    step=batch_id,
                )

                batch = {
                    k: v.to(g_model.device).to(torch.float32) for k, v in batch.items()
                }
                y_sample = g_model.reorder_label_phase(batch).to(g_model.device)

                g_pred = g_model(batch["X"], logits=g_model.logits)
                g_pred = torch.sigmoid(g_pred).to(g_model.device)

                g_pred = g_pred.detach().cpu().numpy()
                y_sample = y_sample.detach().cpu().numpy()
                batch["X"] = batch["X"].detach().cpu().numpy()

                logger.log_hdf5(
                    batch["X"],
                    data_split=args.data_split,
                    data_type="waveform",
                    step=batch_id,
                )
                logger.log_hdf5(
                    y_sample,
                    data_split=args.data_split,
                    data_type="label",
                    step=batch_id,
                )
                logger.log_hdf5(
                    g_pred,
                    data_split=args.data_split,
                    data_type="prediction",
                    step=batch_id,
                )

                pbar.set_postfix()
                pbar.update(batch_size)

    print(f"Inference complete for {args.data_split} split")
