import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm


class Config:
    """Configuration for loss surface visualization"""
    SAMPLING_RATE = 100
    SIGNAL_LENGTH = 3001
    SIGNAL_CENTER = 1500
    SIGMA = 20
    DISPLAY_RANGE_HEIGHT = 3.5
    DISPLAY_RANGE_POSITION = 3.5
    PRED_TIME_CENTER = 0.3
    PRED_HEIGHT = 0.7
    PRED_SIGMA_TIME = 0.2
    PRED_POINT_INTERVAL = 0.1
    PRED_POINT_RANGE_START = 0.0
    PRED_POINT_RANGE_END = 0.8
    DPI = 300


class LossSurfaceGenerator:
    """Generate loss surfaces for optimization visualization"""

    def __init__(self, config: Config):
        self.config = config
        self.loss_fn = torch.nn.BCELoss()

    def gaussian_curve(self, length=None, peak_pos=None, sigma=None, height=1.0):
        """Generate Gaussian curve"""
        length = length or self.config.SIGNAL_LENGTH
        peak_pos = peak_pos or self.config.SIGNAL_CENTER
        sigma = sigma or self.config.SIGMA
        x = np.arange(length)
        curve = norm.pdf(x, loc=peak_pos, scale=sigma)
        return (curve / np.max(curve)) * height

    def generate_height_loss_surface(self):
        """Generate loss surface for height-only optimization"""
        label_curve = self.gaussian_curve()
        time_range = np.linspace(-self.config.DISPLAY_RANGE_HEIGHT, self.config.DISPLAY_RANGE_HEIGHT, 101)
        height_range = np.linspace(0.01, 0.99, 101)
        loss_surface = np.zeros((len(height_range), len(time_range)))

        for i, height in enumerate(height_range):
            for j, time_pos in enumerate(time_range):
                sample_idx = int((time_pos * self.config.SAMPLING_RATE) + self.config.SIGNAL_CENTER)
                if 0 <= sample_idx < self.config.SIGNAL_LENGTH:
                    pred_tensor = torch.tensor([height], dtype=torch.float32)
                    target_tensor = torch.tensor([label_curve[sample_idx]], dtype=torch.float32)
                    loss = self.loss_fn(pred_tensor, target_tensor)
                    loss_surface[i, j] = loss.item()
                else:
                    loss_surface[i, j] = 1.0

        return loss_surface, time_range, height_range

    def generate_gaussian_curve_loss_surface(self):
        """Generate loss surface for Gaussian curve position optimization"""
        label_curve = self.gaussian_curve()
        truncate_range = 60 * self.config.SIGMA
        template_length = 2 * truncate_range + 1
        template_center = template_length // 2
        template_curve = self.gaussian_curve(template_length, template_center, self.config.SIGMA, 1.0)
        position_range_seconds = np.linspace(-self.config.DISPLAY_RANGE_POSITION, self.config.DISPLAY_RANGE_POSITION, 51)
        height_range = np.linspace(0.01, 0.99, 51)
        position_range_samples = (position_range_seconds * self.config.SAMPLING_RATE) + self.config.SIGNAL_CENTER
        loss_surface = np.zeros((len(height_range), len(position_range_seconds)))

        for i, height in enumerate(height_range):
            for j, pos_samples in enumerate(position_range_samples):
                pred_curve = np.zeros(self.config.SIGNAL_LENGTH)
                start_idx = int(pos_samples - truncate_range)
                end_idx = int(pos_samples + truncate_range + 1)
                template_start = max(0, -start_idx)
                template_end = min(template_length, self.config.SIGNAL_LENGTH - start_idx)
                pred_start = max(0, start_idx)
                pred_end = min(self.config.SIGNAL_LENGTH, end_idx)

                if pred_start < pred_end and template_start < template_end:
                    template_slice = template_curve[template_start:template_end] * height
                    pred_curve[pred_start:pred_end] = template_slice

                pred_tensor = torch.tensor(pred_curve, dtype=torch.float32)
                target_tensor = torch.tensor(label_curve, dtype=torch.float32)
                loss = self.loss_fn(pred_tensor, target_tensor)
                loss_surface[i, j] = loss.item()

        return loss_surface, position_range_seconds, height_range


class LossSurfaceVisualizer:
    """Visualizer for loss surfaces"""

    def __init__(self, config: Config):
        self.config = config

    def plot_loss_surfaces(self, loss_generator):
        """Plot both loss surfaces in a single figure with vertical subplots"""
        loss1, time_range, height_range1 = loss_generator.generate_height_loss_surface()
        loss2, pos_range, height_range2 = loss_generator.generate_gaussian_curve_loss_surface()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

        X1, Y1 = np.meshgrid(time_range, height_range1)
        contour1 = ax1.contour(X1, Y1, loss1, levels=10, colors='black', linewidths=0.5)
        ax1.clabel(contour1, inline=True, fontsize=8)
        contourf1 = ax1.contourf(X1, Y1, loss1, levels=50, alpha=0.8, cmap='RdBu_r')
        cbar1 = plt.colorbar(contourf1, ax=ax1)
        cbar1.set_label('BCE Loss')

        label_curve = loss_generator.gaussian_curve()
        ax1_twin = ax1.twinx()
        display_time = (np.arange(self.config.SIGNAL_LENGTH) - self.config.SIGNAL_CENTER) / self.config.SAMPLING_RATE
        mask = (display_time >= -self.config.DISPLAY_RANGE_HEIGHT) & (display_time <= self.config.DISPLAY_RANGE_HEIGHT)
        ax1_twin.plot(display_time[mask], label_curve[mask], 'w-', linewidth=3, alpha=0.8, label='Label Curve')

        pred_time_center = self.config.PRED_TIME_CENTER
        pred_height = self.config.PRED_HEIGHT
        pred_sigma_time = self.config.PRED_SIGMA_TIME
        pred_curve_values = pred_height * np.exp(-0.5 * ((display_time - pred_time_center) / pred_sigma_time) ** 2)
        ax1_twin.plot(display_time[mask], pred_curve_values[mask], '-', color='red', linewidth=2, alpha=0.8, label='Prediction Curve')

        point_times = np.arange(self.config.PRED_POINT_RANGE_START, self.config.PRED_POINT_RANGE_END, self.config.PRED_POINT_INTERVAL)
        point_values = pred_height * np.exp(-0.5 * ((point_times - pred_time_center) / pred_sigma_time) ** 2)
        ax1_twin.plot(point_times, point_values, 'o', color='yellow', markersize=3, markeredgewidth=0, zorder=11)
        ax1_twin.plot([], [], '-', color='yellow', linewidth=2, label='Updates')

        for pt_time, pt_val in zip(point_times, point_values):
            time_idx = np.argmin(np.abs(time_range - pt_time))
            height_idx = np.argmin(np.abs(height_range1 - pt_val))
            if height_idx > 0 and height_idx < len(height_range1) - 1:
                grad_y = (loss1[height_idx + 1, time_idx] - loss1[height_idx - 1, time_idx]) / (2 * (height_range1[1] - height_range1[0]))
                arrow_scale = 0.05
                ax1.arrow(pt_time, pt_val, 0, -grad_y * arrow_scale, head_width=0.03, head_length=0.01, fc='yellow', ec='yellow', linewidth=1, zorder=12)

        ax1_twin.set_yticks([])
        ax1_twin.set_yticklabels([])
        ax1_twin.set_ylim(0, 1.0)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Height')
        ax1.set_ylim(0, 1.0)
        ax1.set_title('BCE Loss Surface - Time Independent')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

        X2, Y2 = np.meshgrid(pos_range, height_range2)
        contour2 = ax2.contour(X2, Y2, loss2, levels=10, colors='black', linewidths=0.5)
        ax2.clabel(contour2, inline=True, fontsize=8)
        contourf2 = ax2.contourf(X2, Y2, loss2, levels=100, alpha=0.8, cmap='RdBu_r')
        cbar2 = plt.colorbar(contourf2, ax=ax2)
        cbar2.set_label('BCE Loss')

        ax2_twin = ax2.twinx()
        truncate_range = 60 * self.config.SIGMA
        truncate_time = min(truncate_range / self.config.SAMPLING_RATE, self.config.DISPLAY_RANGE_POSITION)
        mask = (display_time >= -truncate_time) & (display_time <= truncate_time)
        line_label = 'Label Curve (Truncated)' if truncate_time < self.config.DISPLAY_RANGE_POSITION else 'Label Curve'
        ax2_twin.plot(display_time[mask], label_curve[mask], 'w-', linewidth=3, alpha=0.8, label=line_label)

        pred_mask = (display_time >= -truncate_time) & (display_time <= truncate_time)
        ax2.plot(self.config.PRED_TIME_CENTER, self.config.PRED_HEIGHT, 'o', color='red', markersize=6, markeredgewidth=0, zorder=10)
        pred_peak_pos = self.config.PRED_TIME_CENTER * self.config.SAMPLING_RATE + self.config.SIGNAL_CENTER
        pred_curve_full = loss_generator.gaussian_curve(peak_pos=pred_peak_pos, height=self.config.PRED_HEIGHT)
        ax2_twin.plot(display_time[pred_mask], pred_curve_full[pred_mask], '-', color='red', linewidth=2, alpha=0.8, label='Prediction Curve')

        point_pos = self.config.PRED_TIME_CENTER
        point_height = self.config.PRED_HEIGHT
        pos_idx = np.argmin(np.abs(pos_range - point_pos))
        height_idx = np.argmin(np.abs(height_range2 - point_height))

        if (pos_idx > 0 and pos_idx < len(pos_range) - 1 and height_idx > 0 and height_idx < len(height_range2) - 1):
            grad_x = (loss2[height_idx, pos_idx + 1] - loss2[height_idx, pos_idx - 1]) / (2 * (pos_range[1] - pos_range[0]))
            grad_y = (loss2[height_idx + 1, pos_idx] - loss2[height_idx - 1, pos_idx]) / (2 * (height_range2[1] - height_range2[0]))
            arrow_scale = 1
            ax2.arrow(point_pos, point_height, -grad_x * arrow_scale, -grad_y * arrow_scale, head_width=0.02, head_length=0.05, fc='yellow', ec='yellow', linewidth=2, zorder=12)

            arrow_end_pos = point_pos - grad_x * arrow_scale
            arrow_end_height = point_height - grad_y * arrow_scale
            arrow_end_peak_pos = arrow_end_pos * self.config.SAMPLING_RATE + self.config.SIGNAL_CENTER
            dashed_curve_full = loss_generator.gaussian_curve(peak_pos=arrow_end_peak_pos, height=arrow_end_height)
            ax2_twin.plot(display_time[pred_mask], dashed_curve_full[pred_mask], '--', color='yellow', linewidth=2, alpha=0.6, label='Updates')

        ax2_twin.set_yticks([])
        ax2_twin.set_yticklabels([])
        ax2_twin.set_ylim(0, 1.0)
        ax2.set_xlabel('Peak Position (seconds)')
        ax2.set_ylabel('Peak Height')
        title = f'BCE Loss Surface - Gaussian Curve Peak Position'
        if truncate_time < self.config.DISPLAY_RANGE_POSITION:
            title += f' (Truncated to ±{truncate_time:.2f}s)'
        ax2.set_title(title)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

        plt.tight_layout()
        plt.savefig('loss_surface_combined.png', dpi=self.config.DPI, bbox_inches='tight')

        return (loss1, time_range, height_range1), (loss2, pos_range, height_range2)


def main():
    """Main function to generate and plot loss surfaces"""
    config = Config()
    loss_generator = LossSurfaceGenerator(config)
    visualizer = LossSurfaceVisualizer(config)
    surface_data = visualizer.plot_loss_surfaces(loss_generator)
    return surface_data


if __name__ == "__main__":
    main()

