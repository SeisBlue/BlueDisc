import numpy as np
import seisbench
from seisbench import config
from seisbench.generate.labeling import SupervisedLabeller, gaussian_pick


class TaperedDetectionLabeller(SupervisedLabeller):
    """
    Modified version of DetectionLabeller that adds Gaussian tails to detections.
    """

    def __init__(
        self,
        p_phases,
        s_phases=None,
        shape=None,
        sigma=10,
        factor=1.4,
        fixed_window=None,
        **kwargs,
    ):
        self.label_method = "probabilistic"
        self.label_columns = "detections"
        self.shape = shape
        self.sigma = sigma

        if isinstance(p_phases, str):
            self.p_phases = [p_phases]
        else:
            self.p_phases = p_phases

        if isinstance(s_phases, str):
            self.s_phases = [s_phases]
        elif s_phases is None:
            self.s_phases = []
        else:
            self.s_phases = s_phases

        if s_phases is not None and fixed_window is not None:
            seisbench.logger.warning(
                "Provided both S phases and fixed window length to TaperedDetectionLabeller. "
                "Will use fixed window size and ignore S phases."
            )

        self.factor = factor
        self.fixed_window = fixed_window

        kwargs["dim"] = kwargs.get("dim", -2)
        super().__init__(label_type="multi_class", **kwargs)

    def label(self, X, metadata):
        sample_dim, channel_dim, width_dim = self._get_dimension_order_from_config(
            config, self.ndim
        )

        if self.fixed_window:
            # Only label until end of fixed window
            factor = 0
        else:
            factor = self.factor

        if self.ndim == 2:
            y = np.zeros((1, X.shape[width_dim]))
            p_arrivals = [
                metadata[phase]
                for phase in self.p_phases
                if phase in metadata and not np.isnan(metadata[phase])
            ]
            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase]
                    for phase in self.s_phases
                    if phase in metadata and not np.isnan(metadata[phase])
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrival = min(p_arrivals)
                s_arrival = min(s_arrivals)
                p_to_s = s_arrival - p_arrival
                if s_arrival >= p_arrival:
                    # Only annotate valid options
                    p0 = max(int(p_arrival), 0)
                    p1 = max(int(s_arrival + factor * p_to_s), 0)
                    if self.shape == "gaussian":
                        # Calculate Gaussian tails
                        left_edge = gaussian_pick(p0, X.shape[width_dim], self.sigma)
                        right_edge = gaussian_pick(p1, X.shape[width_dim], self.sigma)

                        y[0, :p0] += left_edge[
                            :p0
                        ]  # Add left tail before detection window
                        y[0, p1:] += right_edge[
                            p1:
                        ]  # Add right tail after detection window

                    y[0, p0:p1] = 1

        elif self.ndim == 3:
            y = np.zeros(
                shape=(
                    X.shape[sample_dim],
                    1,
                    X.shape[width_dim],
                )
            )
            p_arrivals = [
                metadata[phase] for phase in self.p_phases if phase in metadata
            ]

            if self.fixed_window is not None:
                # Fake S arrivals for simulating fixed window
                s_arrivals = [x + self.fixed_window for x in p_arrivals]
            else:
                s_arrivals = [
                    metadata[phase] for phase in self.s_phases if phase in metadata
                ]

            if len(p_arrivals) != 0 and len(s_arrivals) != 0:
                p_arrivals = np.stack(p_arrivals, axis=-1)  # Shape (samples, phases)
                s_arrivals = np.stack(s_arrivals, axis=-1)

                mask = np.logical_and(
                    np.any(~np.isnan(p_arrivals), axis=1),
                    np.any(~np.isnan(s_arrivals), axis=1),
                )
                if not mask.any():
                    return y

                p_arrivals = np.nanmin(
                    p_arrivals[mask, :], axis=1
                )  # Shape (samples (which are present),)
                s_arrivals = np.nanmin(s_arrivals[mask, :], axis=1)
                p_to_s = s_arrivals - p_arrivals

                starts = p_arrivals.astype(int)
                ends = (s_arrivals + factor * p_to_s).astype(int)

                # print(mask, starts, ends)

                # ============================================================
                # The main difference from original:
                # Add gaussian tails to detections
                for i, s, e in zip(np.arange(len(mask))[mask], starts, ends):
                    s = max(0, s)
                    e = max(0, e)
                    if self.shape == "gaussian":
                        # Calculate Gaussian tails
                        left_edge = gaussian_pick(s, e - s, self.sigma)[
                            0:s
                        ]  # Left tail
                        right_edge = gaussian_pick(
                            e, X.shape[width_dim] - e, self.sigma
                        )  # Right tail

                        # Set detection window and Gaussian tails
                        y[i, 0, max(0, s - len(left_edge)) : s] = (
                            left_edge  # Left Gaussian
                        )
                        y[i, 0, e : min(e + len(right_edge), X.shape[width_dim])] = (
                            right_edge  # Right Gaussian
                        )

                    y[i, 0, s:e] = 1
                # ============================================================

        else:
            raise ValueError(
                f"Illegal number of input dimensions for DetectionLabeller (ndim={self.ndim})."
            )

        return y

    def __str__(self):
        return f"DetectionLabeller (label_type={self.label_type}, dim={self.dim})"
