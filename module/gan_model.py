import os
import glob
import torch
from tqdm import tqdm

from module.discriminator import DBuilder
from module.generator import GBuilder

bce_logits = torch.nn.BCEWithLogitsLoss()


def sgan_d_loss(D_real, D_fake):
    real_loss = bce_logits(D_real, torch.ones_like(D_real))
    fake_loss = bce_logits(D_fake, torch.zeros_like(D_fake))
    return (real_loss + fake_loss) / 2


def sgan_g_loss(D_real, D_fake):
    return bce_logits(D_fake, torch.ones_like(D_fake))


class GANModel(torch.nn.Module):
    def __init__(
        self,
        gan_type=None,
        generator=None,
        discriminator=None,
        g_data_weight=1.0,
        gan_loss_weight=1.0,
        logger=None,
    ):
        super(GANModel, self).__init__()
        self.gan_type = gan_type
        self.g_model = generator
        self.d_model = discriminator

        self.g_data_weight = g_data_weight
        self.gan_loss_weight = gan_loss_weight
        self.logger = logger

        self.steps = 0
        self.epoch = 1
        self.bce_logits = torch.nn.BCEWithLogitsLoss()

        self.d_gan_loss = None
        self.g_gan_loss = None

        if gan_type:
            self.d_gan_loss = sgan_d_loss
            self.g_gan_loss = sgan_g_loss

    def compute_d_gan_loss(self, batch, y, g_pred):
        gan = self.gan_type
        d_real = self.d_model(y, condition=batch["X"])
        d_fake = self.d_model(g_pred, condition=batch["X"])
        d_gan_loss = self.d_gan_loss(d_real, d_fake)

        if self.logger:
            self.logger.log_metric(
                f"d_{gan}_real", d_real.mean().item(), step=self.steps
            )
            self.logger.log_metric(
                f"d_{gan}_fake", d_fake.mean().item(), step=self.steps
            )
            self.logger.log_metric(f"d_{gan}_loss", d_gan_loss.item(), step=self.steps)

        return d_gan_loss


    def compute_d_loss(self, batch, y, train=True):
        g_pred = self.g_model(batch["X"], logits=self.g_model.logits).detach()
        if self.g_model.logits:
            g_pred = torch.sigmoid(g_pred).to(self.g_model.device)

        d_gan_loss = self.compute_d_gan_loss(batch, y, g_pred)

        d_loss = d_gan_loss

        if self.logger:
            self.logger.log_metric("d_loss", d_loss.item(), step=self.steps)

        if train:
            self.d_model.optimizer.zero_grad()
            d_loss.backward()
            self.d_model.optimizer.step()

        return d_loss

    def compute_g_gan_loss(self, batch, y, steps):
        g_pred = self.g_model(batch["X"], logits=True)
        if self.g_model.logits:
            g_pred = torch.sigmoid(g_pred).to(self.g_model.device)
        gan = self.gan_type
        g_real = self.d_model(y, condition=batch["X"])
        g_fake = self.d_model(g_pred, condition=batch["X"])
        g_gan_loss = self.g_gan_loss(g_real, g_fake)

        if self.logger:
            self.logger.log_metric(
                f"g_{gan}_real",
                g_real.mean().item(),
                step=steps,
            )
            self.logger.log_metric(
                f"g_{gan}_fake",
                g_fake.mean().item(),
                step=steps,
            )
            self.logger.log_metric(f"g_{gan}_loss", g_gan_loss.item(), step=steps)
        return g_gan_loss

    def compute_g_loss(self, batch, y, train=True):
        g_gan_loss = 0
        if self.d_model:
            g_gan_loss = self.compute_g_gan_loss(batch, y, self.steps)

        g_pred = self.g_model(batch["X"], logits=self.g_model.logits)

        g_data_loss = self.bce_logits(g_pred, y)
        self.logger.log_metric("g_data_loss", g_data_loss.item(), step=self.steps)

        g_loss = (
            g_gan_loss * self.gan_loss_weight + g_data_loss * self.g_data_weight
        ) / (self.gan_loss_weight + self.g_data_weight)
        self.logger.log_metric("g_loss", g_loss.item(), step=self.steps)


        if train:
            self.g_model.optimizer.zero_grad()
            g_loss.backward()
            self.g_model.optimizer.step()

        return g_loss, g_data_loss

    def train_loop(self, dataloader, test_samples, max_steps):
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        with tqdm(total=size, desc="Training", ncols=80) as pbar:
            torch.autograd.set_detect_anomaly(True)
            for batch_id, batch in enumerate(dataloader):
                batch.pop("trace_name", "")

                batch = {
                    k: v.to(device=self.g_model.device, dtype=torch.float32)
                    for k, v in batch.items()
                }
                y = self.g_model.reorder_label_phase(batch)

                self.g_model.train()
                if self.d_model:
                    self.d_model.train()

                if self.d_model:
                    d_loss = self.compute_d_loss(batch, y, train=True)

                g_loss, train_loss = self.compute_g_loss(batch, y, train=True)

                # Test and log samples
                eval_loss = self.sample_testing(test_samples)

                pbar.set_postfix(t=f"{train_loss:>4e}", e=f"{eval_loss:>4e}")
                pbar.update(batch_size)

                # save checkpoint every step
                checkpoint = f"g_model_epoch_{self.epoch:0>3}_step_{self.steps:0>7}.pt"
                checkpoint_path = os.path.join(
                    self.logger.base_path, "checkpoint", checkpoint
                )
                self.save_checkpoint(checkpoint_path)

                self.steps += 1
                if self.steps >= max_steps:
                    break

    def sample_testing(self, test_samples):
        self.g_model.eval()
        if self.d_model:
            self.d_model.eval()

        with torch.no_grad():
            test_samples = {
                k: v.to(device=self.g_model.device, dtype=torch.float32) for k, v in test_samples.items()
            }
            y_sample = self.g_model.reorder_label_phase(test_samples).to(
                self.g_model.device
            )

            sample_pred = self.g_model(test_samples["X"], logits=self.g_model.logits)
            sample_loss = self.g_model.batch_loss_fn(sample_pred, y_sample)

            eval_loss = sample_loss.mean()

            if self.g_model.logits:
                sample_pred = torch.sigmoid(sample_pred).to(self.g_model.device)

            if self.d_model:
                sample_D_real = self.d_model(y_sample, condition=test_samples["X"])
                sample_D_fake = self.d_model(sample_pred, condition=test_samples["X"])

                sample_D_real = sample_D_real.detach().cpu().numpy()
                sample_D_fake = sample_D_fake.detach().cpu().numpy()

                self.logger.log_samples_metrics(sample_D_real, "Dreal", self.steps)
                self.logger.log_samples_metrics(sample_D_fake, "Dfake", self.steps)

            sample_pred = sample_pred.detach().cpu().numpy()
            sample_loss = sample_loss.detach().cpu().numpy()
            eval_loss = eval_loss.detach().cpu().numpy()

            self.logger.log_hdf5(
                sample_pred, data_split="track", data_type="prediction", step=self.steps
            )
            self.logger.log_samples_metrics(sample_loss, "loss", self.steps)
            self.logger.log_metric("eval_loss", eval_loss, self.steps)

        return eval_loss

    def fit(self, train_loader, test_samples, max_steps=1000):
        print(f"\ncurrent steps: {self.steps}/{max_steps}")
        while self.steps < max_steps:
            print(f"\nEpoch {self.epoch}\n-------------------------------")
            self.train_loop(train_loader, test_samples, max_steps)
            print(f"\ncurrent steps: {self.steps}/{max_steps}")

            self.epoch += 1
            if self.steps >= max_steps:
                break

    def save_checkpoint(self, path):
        """Save model checkpoint states including optimizers"""
        states = {
            "metadata": {
                "current_steps": self.steps,
                "current_epoch": self.epoch,
                "phase_order": self.g_model.phase_order,
                "label": self.g_model.phase_order[-1],
                "gan_type": self.gan_type,
                "g_data_weight": self.g_data_weight,
                "gan_loss_weight": self.gan_loss_weight,
                "g_model": self.g_model.name,
                "g_optimizer": self.g_model.optimizer.__class__.__name__,
                "g_lr": self.g_model.optimizer.param_groups[0]["lr"],
                "d_model": self.d_model.name if self.d_model else None,
                "d_optimizer": (
                    self.d_model.optimizer.__class__.__name__ if self.d_model else None
                ),
                "d_lr": (
                    self.d_model.optimizer.param_groups[0]["lr"]
                    if self.d_model
                    else None
                ),
            },
            "g_model": self.g_model.state_dict(),
            "g_optimizer": self.g_model.optimizer.state_dict(),
        }
        if self.g_model.scheduler:
            states["g_scheduler"] = self.g_model.scheduler.state_dict()

        if self.d_model:
            states["d_model"] = self.d_model.state_dict()
            states["d_optimizer"] = self.d_model.optimizer.state_dict()
            if self.d_model.scheduler:
                states["d_scheduler"] = self.d_model.scheduler.state_dict()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(states, path)

    def load_checkpoint(self, path):
        """Load model checkpoint states including optimizers"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file {path} does not exist.")

        checkpoint = torch.load(path)
        metadata = checkpoint["metadata"]
        self.steps = metadata["current_steps"]
        self.epoch = metadata["current_epoch"]
        self.gan_type = metadata["gan_type"]
        self.g_data_weight = metadata["g_data_weight"]
        self.gan_loss_weight = metadata["gan_loss_weight"]

        self.g_model = GBuilder().build(
            metadata["g_model"], metadata["label"], metadata["g_lr"]
        )
        self.g_model.load_state_dict(checkpoint["g_model"])
        self.g_model.optimizer.load_state_dict(checkpoint["g_optimizer"])

        if metadata["d_model"]:
            self.d_model = DBuilder().build(metadata["d_model"], metadata["d_lr"])
            self.d_model.load_state_dict(checkpoint["d_model"])
            self.d_model.optimizer.load_state_dict(checkpoint["d_optimizer"])
        else:
            self.d_model = None

        print(f"Checkpoint loaded from {path}")
        print(metadata)
        return self

    def load_checkpoint_by_dir(self, checkpoint_dir, step=None, epoch=None):
        """
        Load order: Step > Epoch > Latest (if both == none)
        """
        if step is not None:
            pattern = os.path.join(
                checkpoint_dir, f"g_model_epoch_*_step_{step:0>7}.pt"
            )
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint found for step {step} in {checkpoint_dir}"
                )

        elif epoch is not None:
            pattern = os.path.join(
                checkpoint_dir, f"g_model_epoch_{epoch:0>3}_step_*.pt"
            )
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint found for epoch {epoch} in {checkpoint_dir}"
                )

        else:
            pattern = os.path.join(checkpoint_dir, f"g_model_epoch_*_step_*.pt")
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        checkpoint = os.path.basename(max(checkpoints))
        self.load_checkpoint(os.path.join(checkpoint_dir, checkpoint))
        return checkpoint
