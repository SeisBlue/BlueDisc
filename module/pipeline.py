import re

import numpy as np
import seisbench.generate as sbg

from module.labeler import TaperedDetectionLabeller


class AugmentationsBuilder:
    def __init__(self, dataset, label_type="D"):
        self.dataset = dataset
        self.label_type = label_type
        self.phase_dict, self.p_phases, self.s_phases = self.get_phase()

        detection_labeller = TaperedDetectionLabeller(
            self.p_phases,
            s_phases=self.s_phases,
            shape="gaussian",
            sigma=20,
            key=("X", "detections"),
        )

        self.augmentations = [
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(self.phase_dict.keys()),
                        samples_before=3000,
                        windowlen=6000,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                windowlen=3001,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(
                label_columns=self.phase_dict, model_labels="PSN", sigma=20, dim=0
            ),
            detection_labeller,
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            self.add_trace_name,
        ]

    def get_phase(self):
        metadata = self.dataset.metadata
        phase_dict = {}
        # Use a regular expression to match phase labels in column names, e.g., trace_P_arrival_sample
        pattern = re.compile(r"trace_([a-zA-Z0-9]+)_arrival_sample")
        for col in metadata.columns:
            match = pattern.match(col)
            if match:
                phase = match.group(1)
                if "P" in phase or "p" in phase:
                    phase_dict[col] = "P"
                elif "S" in phase or "s" in phase:
                    phase_dict[col] = "S"

        p_phases = [key for key, val in phase_dict.items() if val == "P"]
        s_phases = [key for key, val in phase_dict.items() if val == "S"]

        return phase_dict, p_phases, s_phases

    @staticmethod
    def add_trace_name(state_dict):
        x, metadata = state_dict["X"]
        state_dict["trace_name"] = (metadata["trace_name"], metadata)

    def build(self):
        return self.augmentations


if __name__ == "__main__":
    import seisbench.data as sbd

    data = sbd.ETHZ()
    builder = AugmentationsBuilder(data)
    phase_dict, p_phases, s_phases = builder.get_phase()
    print(data.metadata.columns)
    print(phase_dict)
    print(p_phases)
    print(s_phases)
