import seisbench.models as sbm
import torch


class BaseGenerator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.cuda()

        self.phase_order = "PSN"
        self.current_steps = 0
        self.current_epoch = 1
        self.optimizer = None
        self.scheduler = None
        self.logits = True
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def batch_loss_fn(pred, y):
        return (
            torch.nn.BCEWithLogitsLoss(reduction="none")(pred, y)
            .mean(dim=-1)
            .mean(dim=-1)
        )

    def reorder_label_phase(self, batch):
        labels = {
            "P": batch["y"][:, 0],
            "S": batch["y"][:, 1],
            "N": batch["y"][:, 2],
            "D": batch["detections"][:, 0],
        }

        return torch.stack([labels[phase] for phase in self.phase_order], dim=1)


class WrappedPhaseNet(BaseGenerator, sbm.PhaseNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__class__.__name__ = "WrappedPhaseNet"
        self.__name__ = "PhaseNet"

        self.phase_order = "PSN"
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def batch_loss_fn(pred, y):
        return (
            torch.nn.BCEWithLogitsLoss(reduction="none")(pred, y)
            .mean(dim=-1)
            .mean(dim=-1)
        )



class GBuilder:
    def __init__(self):
        self.model_name_dict = {
            "PN": "WrappedPhaseNet",
        }
        self.model_dict = {
            "WrappedPhaseNet": {
                "model": WrappedPhaseNet,
                "model_params": {
                    "phases": "PSN",
                    "norm": "peak",
                },
                "lr": 1e-3,
            },
        }

    def build(self, g_model_name, label, learning_rate):
        g_model_name = self.model_name_dict[g_model_name]
        g_model = self.model_dict[g_model_name]["model"]
        g_model = g_model(**self.model_dict[g_model_name]["model_params"])
        g_model.phase_order = "PS" + label

        if not learning_rate:
            learning_rate = self.model_dict[g_model_name]["lr"]

        g_model.lr = learning_rate

        g_model.optimizer = torch.optim.Adam(
            g_model.parameters(),
            lr=learning_rate,
            betas=(0.0, 0.999),
        )
        return g_model


if __name__ == "__main__":
    for model in [
        WrappedPhaseNet(phases="PSN", norm="peak"),
    ]:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters for {model.__name__}: {total_params}")
