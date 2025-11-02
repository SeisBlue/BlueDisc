import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    CGAN Discriminator BlueDisc
    """

    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()
        self.name = "BlueDisc"
        self.x_condition_model = self.model(in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(47616, 1)
        self.optimizer = None
        self.scheduler = None

    @staticmethod
    def model(in_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=11, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 64, kernel_size=11, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, label, condition):
        # Ensure condition is on the same device as label
        condition = condition.to(label.device)
        x = torch.cat([label, condition], dim=1)
        x = self.x_condition_model(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DBuilder:
    def __init__(self):
        self.model_dict = {
            "BlueDisc": {"model": Discriminator, "lr": 1e-3},
        }

    def build(self, d_model_name, learning_rate):
        # Use proper device detection: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        d_model = Discriminator().to(device)

        if not learning_rate:
            learning_rate = self.model_dict[d_model_name]["lr"]

        d_model.lr = learning_rate
        d_model.optimizer = torch.optim.Adam(
            d_model.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.999),
        )

        d_model.device = device

        return d_model


if __name__ == "__main__":
    for model_class in [Discriminator]:
        model = model_class()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters for {model_class.__name__}: {total_params}")

