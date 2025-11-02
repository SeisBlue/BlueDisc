import argparse
import os

import mlflow.pytorch
import seisbench.data as sbd
import seisbench.generate as sbg
import torch
from torch.utils.data import DataLoader
from module.discriminator import DBuilder
from module.gan_model import GANModel
from module.generator import GBuilder
from module.logger import MLFlowLogger
from module.device_manager import DeviceManager
from module.pipeline import AugmentationsBuilder
from module.random_seed import RandomSeedManager

# Ensure web_host and web_port are initialized at the start
web_host = "0.0.0.0"
web_port = 8080

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, choices=["D", "N"])
    parser.add_argument("--g-model", type=str, required=True)
    parser.add_argument("--d-model", type=str)
    parser.add_argument("--gan-type", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--g-lr", type=float)
    parser.add_argument("--d-lr", type=float)
    parser.add_argument("--data-weight", type=float, default=1.0)
    parser.add_argument("--gan-weight", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--plot-sample", type=int)

    args = parser.parse_args()

    seed_value = 42
    seed_manager = RandomSeedManager(seed_value)
    seed_manager.set_seed()

    # 初始化設備管理器
    device_manager = DeviceManager(args.device)

    sample_size = args.sample_size
    batch_size = args.batch_size
    max_steps = args.max_steps

    gan_loss_weight = args.gan_weight
    if not args.gan_type:
        gan_loss_weight = 0
        args.data_weight = 1  # 沒有 GAN 條件下強制設定 data_weight 為 1

    # 構建生成器和判別器模型
    g_builder = GBuilder()
    g_model = g_builder.build(args.g_model, args.label, args.g_lr)

    d_model = None

    if args.d_model:
        d_builder = DBuilder()
        d_model = d_builder.build(args.d_model, args.d_lr)

    project_root = os.getcwd()

    print("Using local MLflow configuration")
    mlflow_host = "127.0.0.1"
    mlflow_port = 5000

    mlflow.set_tracking_uri(f"http://{mlflow_host}:{mlflow_port}")

    # 設定實驗名稱
    experient_name = f"{args.g_model}_{args.label}"

    if d_model:
        experient_name += f"_{d_model.name}"
    if args.gan_type:
        experient_name += f"_{args.gan_type}"
    if args.data_weight == 0 and args.gan_type:
        experient_name += "_Data0"
    elif args.gan_type:
        experient_name += f"_Data{args.data_weight}"

    mlflow.set_experiment(experient_name)
    print(f"Experient: {experient_name}")

    # 動態載入資料集
    data_class = getattr(sbd, args.dataset)
    data = data_class(sampling_rate=100)

    train, dev, test = data.train_dev_test()
    print(
        f"train: {len(train)}, dev: {len(dev)}, track: {sample_size}, test: {len(test)}"
    )

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)

    aug_builder = AugmentationsBuilder(dataset=data, label_type=args.label)
    augmentations = aug_builder.build()

    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)

    g = torch.Generator()
    g.manual_seed(seed_value)

    if args.batch_size is None:
        raise ValueError("Batch size must be specified and should be an integer.")

    num_workers = os.cpu_count() or 1  # Ensure num_workers defaults to 1 if os.cpu_count() returns None

    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_manager.worker_init_fn,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        generator=g,
    )

    dev_loader = DataLoader(
        dev_generator,
        batch_size=sample_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_manager.worker_init_fn,
        pin_memory=True,
        generator=g,
    )

    # 取得 dev set 的 sample data
    test_samples = next(iter(dev_loader))
    trace_name = test_samples.pop("trace_name", "")
    label = g_model.reorder_label_phase(test_samples)

    with mlflow.start_run():
        current_run = mlflow.active_run()
        run_id = current_run.info.run_id

        # Initialize logger with the run_id
        logger = MLFlowLogger(
            run_id=run_id,
            mlflow_host=mlflow_host,
            mlflow_port=mlflow_port
        )

        # Initialize GANModel with logger
        gan_model = GANModel(
            gan_type=args.gan_type,
            generator=g_model,
            discriminator=d_model,
            g_data_weight=args.data_weight,
            gan_loss_weight=gan_loss_weight,
            logger=logger,
        )
        print("compile model")
        gan_model = torch.compile(gan_model)

        # Move model to device
        gan_model = device_manager.move_to_device(gan_model)
        print(f"Model moved to device: {device_manager.device}")

        # Log parameters
        logger.log_param("gan_type", args.gan_type)
        logger.log_param("g_model", g_model.name)
        logger.log_param("d_model", None if not d_model else d_model.name)
        logger.log_param("dataset", args.dataset)
        logger.log_param("sample_size", sample_size)
        logger.log_param("batch_size", batch_size)
        logger.log_param("g_lr", g_model.lr)
        logger.log_param("d_lr", None if not d_model else d_model.lr)
        logger.log_param("g_data_weight", args.data_weight)
        logger.log_param("gan_loss_weight", gan_loss_weight)
        logger.log_param("max_steps", max_steps)

        # Log sample data to artifacts (synchronously)
        logger.log_hdf5(
            test_samples["X"].numpy(), data_split="track", data_type="waveform", step=0
        )
        logger.log_hdf5(
            label.numpy(), data_split="track", data_type="label", step=0
        )
        logger.log_text(
            trace_name, data_split="track", data_type="trace_name", step=0
        )

        # Start training
        gan_model.fit(
            train_loader=train_loader,
            test_samples=test_samples,
            max_steps=max_steps,
        )
