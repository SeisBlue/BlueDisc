import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm, gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Config:
    """Configuration for BCE loss experiments with skewed target distribution"""
    SAMPLING_RATE = 100
    SIGNAL_LENGTH = 3001
    SIGNAL_CENTER = 1500
    SIGMA = 20

    LEARNING_RATE = 0.01
    NUM_STEPS = 301
    ADAM_BETA1 = 0.0
    ADAM_BETA2 = 0.9

    TARGET_SKEW = -0
    TARGET_SKEW_END = -0
    TARGET_LOC = 0.0
    TARGET_LOC_END = 0.0
    TARGET_SCALE = 0.0
    TARGET_SCALE_END = 0.0
    TARGET_BOUNDS = (-15, 15)
    RANDOM_SEED = 42

    X_MIN, X_MAX = -3, 3
    NUM_DISPLAY_LINES = 11
    FIGURE_SIZE = (6, 7)
    DPI = 300


class SkewedTargetGenerator:
    """Generate target sequences using skewed normal distribution"""

    def __init__(self, config: Config):
        self.config = config

    def generate_skewed_targets(self, num_steps: int):
        """Generate target positions using skewed normal distribution"""
        np.random.seed(self.config.RANDOM_SEED)
        torch.manual_seed(self.config.RANDOM_SEED)

        targets = []
        positions = []

        for step in range(num_steps):
            progress = step / (num_steps - 1)
            current_scale = self.config.TARGET_SCALE * (1 - progress) + self.config.TARGET_SCALE_END * progress
            current_skew = self.config.TARGET_SKEW * (1 - progress) + self.config.TARGET_SKEW_END * progress
            current_loc = self.config.TARGET_LOC * (1 - progress) + self.config.TARGET_LOC_END * progress

            new_pos = skewnorm.rvs(a=current_skew, loc=current_loc, scale=current_scale)

            new_pos_tensor = torch.tensor(new_pos, dtype=torch.float32)
            bounds_tensor = torch.tensor(self.config.TARGET_BOUNDS, dtype=torch.float32)
            new_pos = torch.clamp(new_pos_tensor, bounds_tensor[0], bounds_tensor[1]).item()
            positions.append(new_pos)

        positions_array = np.array(positions)
        mean_position = np.mean(positions_array)
        centered_positions = positions_array - mean_position
        centered_positions = np.clip(centered_positions, self.config.TARGET_BOUNDS[0], self.config.TARGET_BOUNDS[1])
        positions = centered_positions.tolist()

        for pos in positions:
            target = self._generate_gaussian_curve(pos, height=1.0)
            targets.append(target)

        return torch.stack(targets), positions

    def _generate_gaussian_curve(self, pos_seconds, height):
        """Generate a Gaussian curve at given position and height with fixed sigma"""
        x_indices = torch.arange(self.config.SIGNAL_LENGTH, dtype=torch.float32)
        pos_samples = pos_seconds * self.config.SAMPLING_RATE + self.config.SIGNAL_CENTER
        sigma = self.config.SIGMA
        curve = torch.exp(-0.5 * ((x_indices - pos_samples) / sigma) ** 2)
        curve = curve / curve.max() * height
        return curve


class StandardBCEOptimizer:
    """Standard optimization - directly optimize signal values"""

    def __init__(self, config: Config):
        self.config = config
        self.criterion = torch.nn.BCELoss()

    def optimize_to_targets(self, targets):
        """Optimize signal directly to match moving targets"""
        num_steps = len(targets)
        time_points = torch.linspace(self.config.X_MIN, self.config.X_MAX, 100)
        num_points = len(time_points)

        trajectories = []
        for i, time_point in enumerate(time_points):
            trajectories.append({'time': float(time_point.item()), 'heights': []})

        point_params = torch.zeros(num_points, requires_grad=True)
        with torch.no_grad():
            point_params.data.uniform_(0.01, 0.05)

        optimizer = optim.Adam([point_params], lr=self.config.LEARNING_RATE,
                              betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))

        for step in range(num_steps):
            optimizer.zero_grad()
            current_heights = torch.sigmoid(point_params)

            target_heights = torch.zeros(num_points)
            for i, time_point in enumerate(time_points):
                sample_idx = int((time_point * self.config.SAMPLING_RATE) + self.config.SIGNAL_CENTER)
                if 0 <= sample_idx < self.config.SIGNAL_LENGTH:
                    target_heights[i] = targets[step][sample_idx].item()

            loss = self.criterion(current_heights, target_heights)
            loss.backward()
            optimizer.step()

            for i, traj in enumerate(trajectories):
                traj['heights'].append(current_heights[i].item())

        return trajectories


class GaussianBCEOptimizer:
    """Gaussian-constrained optimization"""

    def __init__(self, config: Config):
        self.config = config
        self.criterion = torch.nn.BCELoss()
        self.x_indices = torch.arange(config.SIGNAL_LENGTH, dtype=torch.float32)

    def optimize_to_targets(self, targets):
        """Optimize Gaussian parameters to match moving targets"""
        num_steps = len(targets)

        pos_param = torch.tensor([0.0], requires_grad=True)
        height_param = torch.tensor([0.0], requires_grad=True)

        optimizer = optim.Adam([pos_param, height_param], lr=self.config.LEARNING_RATE,
                              betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))

        trajectory_pos = []
        trajectory_height = []

        for step in range(num_steps):
            optimizer.zero_grad()

            pos_clamped = torch.clamp(pos_param, self.config.TARGET_BOUNDS[0], self.config.TARGET_BOUNDS[1])
            height_clamped = torch.sigmoid(height_param)

            pos_samples = pos_clamped * self.config.SAMPLING_RATE + self.config.SIGNAL_CENTER
            gaussian_curve = torch.exp(-0.5 * ((self.x_indices - pos_samples) / self.config.SIGMA) ** 2)
            current_signal = gaussian_curve / gaussian_curve.max() * height_clamped

            target = targets[step]
            loss = self.criterion(current_signal, target)

            loss.backward()
            optimizer.step()

            trajectory_pos.append(pos_clamped.item())
            trajectory_height.append(height_clamped.item())

        return trajectory_pos, trajectory_height


class BCEVisualizerThreePlots:
    """Visualizer for three-plot style"""

    def __init__(self, config: Config):
        self.config = config

    def create_three_plot_visualization(self, targets, target_positions,
                                      standard_trajectories, gaussian_pos, gaussian_height):
        """Create three plots: distribution, standard, gaussian"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.config.FIGURE_SIZE)

        display_time = (np.arange(self.config.SIGNAL_LENGTH) - self.config.SIGNAL_CENTER) / self.config.SAMPLING_RATE
        mask = (display_time >= self.config.X_MIN) & (display_time <= self.config.X_MAX)
        x_kde = np.linspace(self.config.X_MIN, self.config.X_MAX, 200)

        y_positions = np.linspace(0, 1.0, len(target_positions))
        target_colors = plt.cm.get_cmap('Greens')(np.linspace(0.3, 1.0, len(target_positions)))

        for i, (x_pos, y_pos, color) in enumerate(zip(target_positions, y_positions, target_colors)):
            ax1.plot([x_pos], [y_pos], 'o', color=color, markersize=3, alpha=0.7,
                    markeredgewidth=0, zorder=3)

        try:
            kde = gaussian_kde(target_positions)
            kde_values = kde(x_kde)
            kde_scaled = kde_values / np.max(kde_values)
            ax1.fill_between(x_kde, 0, kde_scaled, alpha=0.6, color='lightgray',
                           label='Target Distribution', zorder=1)
            ax1.plot(x_kde, kde_scaled, color='black', linewidth=1.0, alpha=0.8, zorder=2)
        except:
            pass

        sm1 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Greens'),
                                   norm=plt.Normalize(vmin=0, vmax=len(target_positions)-1))
        sm1.set_array([])
        cax1 = inset_axes(ax1, width="28%", height="3.5%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.05, 1, 1), bbox_transform=ax1.transAxes)
        cbar1 = plt.colorbar(sm1, cax=cax1, orientation='horizontal', label='Step')
        cbar1.ax.tick_params(labelsize=8)
        cbar1.set_label('Step', fontsize=5)

        ax1.plot([], [], 'o', color='darkgreen', markersize=3, alpha=0.7,
                markeredgewidth=0, label='Target Path')

        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Density / Path Level')
        ax1.set_title(f'Target Distribution, Scale: {self.config.TARGET_SCALE}, Skew: {self.config.TARGET_SKEW}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        ax1.set_xlim(self.config.X_MIN, self.config.X_MAX)
        ax1.legend(loc='lower right')

        steps_to_show = np.linspace(0, len(standard_trajectories[0]['heights']) - 1,
                                   self.config.NUM_DISPLAY_LINES, dtype=int)
        colors = plt.cm.get_cmap('Blues')(np.linspace(0.3, 1.0, self.config.NUM_DISPLAY_LINES))

        for i, (step_idx, color) in enumerate(zip(steps_to_show, colors)):
            curve_times = [traj['time'] for traj in standard_trajectories]
            curve_heights = [traj['heights'][step_idx] for traj in standard_trajectories]

            sorted_data = sorted(zip(curve_times, curve_heights))
            sorted_times, sorted_heights = zip(*sorted_data)

            ax2.plot(sorted_times, sorted_heights, 'o-', color=color,
                    markersize=2, linewidth=0.8, alpha=0.8)

        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Blues'),
                                  norm=plt.Normalize(vmin=0, vmax=max(steps_to_show)))
        sm2.set_array([])
        cax2 = inset_axes(ax2, width="28%", height="3.5%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.05, 1, 1), bbox_transform=ax2.transAxes)
        cbar2 = plt.colorbar(sm2, cax=cax2, orientation='horizontal', label='Step')
        cbar2.ax.tick_params(labelsize=8)
        cbar2.set_label('Step', fontsize=5)

        ax2.plot([], [], 'o-', color='steelblue', markersize=2, linewidth=0.8, alpha=0.8,
                label='Point-wise BCE Optimization')

        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Point-wize BCE Optimization')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        ax2.set_xlim(self.config.X_MIN, self.config.X_MAX)
        ax2.legend(loc='upper left')

        truncate_range = 60 * self.config.SIGMA
        template_length = 2 * truncate_range + 1
        template_center = template_length // 2
        template_curve = self._gaussian_curve(template_length, template_center, self.config.SIGMA, 1.0)

        steps_to_show3 = np.linspace(0, len(gaussian_pos) - 1, self.config.NUM_DISPLAY_LINES, dtype=int)
        colors = plt.cm.get_cmap('Reds')(np.linspace(0.3, 1.0, self.config.NUM_DISPLAY_LINES))

        display_time = (np.arange(self.config.SIGNAL_LENGTH) - self.config.SIGNAL_CENTER) / self.config.SAMPLING_RATE
        mask = (display_time >= self.config.X_MIN) & (display_time <= self.config.X_MAX)

        for idx, (step_idx, color) in enumerate(zip(steps_to_show3, colors)):
            pos = gaussian_pos[step_idx]
            height = gaussian_height[step_idx]

            pred_curve = np.zeros(self.config.SIGNAL_LENGTH)
            pos_samples = pos * self.config.SAMPLING_RATE + self.config.SIGNAL_CENTER

            start_idx = int(pos_samples - truncate_range)
            end_idx = int(pos_samples + truncate_range + 1)

            template_start = max(0, -start_idx)
            template_end = min(template_length, self.config.SIGNAL_LENGTH - start_idx)
            pred_start = max(0, start_idx)
            pred_end = min(self.config.SIGNAL_LENGTH, end_idx)

            if pred_start < pred_end and template_start < template_end:
                template_slice = template_curve[template_start:template_end] * height
                pred_curve[pred_start:pred_end] = template_slice

            ax3.plot(display_time[mask], pred_curve[mask], color=color, linewidth=1.0, alpha=0.8)

        sm3 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Reds'),
                                   norm=plt.Normalize(vmin=0, vmax=max(steps_to_show3)))
        sm3.set_array([])
        cax3 = inset_axes(ax3, width="28%", height="3.5%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.05, 1, 1), bbox_transform=ax3.transAxes)
        cbar3 = plt.colorbar(sm3, cax=cax3, orientation='horizontal', label='Step')
        cbar3.ax.tick_params(labelsize=8)
        cbar3.set_label('Step', fontsize=5)

        ax3.plot([], [], '-', color='crimson', linewidth=1.2, alpha=0.8,
                label='Gaussian-Constrained BCE Evolution')

        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Gaussian-Constrained BCE Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        ax3.set_xlim(self.config.X_MIN, self.config.X_MAX)
        ax3.legend(loc='upper left')

        plt.tight_layout()
        filename = 'bce_loss_three_plots.png'
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')

        return filename

    def _gaussian_curve(self, length, peak_pos, sigma, height=1.0):
        """Generate Gaussian curve for visualization"""
        x = np.arange(length)
        curve = np.exp(-0.5 * ((x - peak_pos) / sigma) ** 2)
        return (curve / np.max(curve)) * height


class BCEVisualizerGridPlots:
    """Visualizer for grid of parameter combinations"""

    def __init__(self, config: Config):
        self.config = config

    def create_grid_visualization(self, scale_values, skew_values):
        """Create a grid of experiments with different scale and skew values"""
        n_scales = len(scale_values)
        n_skews = len(skew_values)

        total_rows = n_skews * 3 + (n_skews - 1)
        fig = plt.figure(figsize=(18, 10))

        height_ratios = []
        for i in range(n_skews):
            height_ratios.extend([1, 1, 1])
            if i < n_skews - 1:
                height_ratios.append(0.7)

        gs = fig.add_gridspec(total_rows, n_scales, height_ratios=height_ratios, hspace=0.1, wspace=0.15)

        for row_idx, skew_val in enumerate(skew_values):
            for col_idx, scale_val in enumerate(scale_values):
                temp_config = Config()
                temp_config.TARGET_SCALE = scale_val
                temp_config.TARGET_SCALE_END = scale_val
                temp_config.TARGET_SKEW = skew_val
                temp_config.TARGET_SKEW_END = skew_val

                target_generator = SkewedTargetGenerator(temp_config)
                standard_optimizer = StandardBCEOptimizer(temp_config)
                gaussian_optimizer = GaussianBCEOptimizer(temp_config)

                targets, target_positions = target_generator.generate_skewed_targets(temp_config.NUM_STEPS)
                standard_trajectories = standard_optimizer.optimize_to_targets(targets)
                gaussian_pos, gaussian_height = gaussian_optimizer.optimize_to_targets(targets)

                base_row = row_idx * 4 if row_idx > 0 else 0

                ax1 = fig.add_subplot(gs[base_row, col_idx])
                ax2 = fig.add_subplot(gs[base_row + 1, col_idx], sharex=ax1)
                ax3 = fig.add_subplot(gs[base_row + 2, col_idx], sharex=ax1)

                if row_idx < n_skews - 1:
                    spacer_ax = fig.add_subplot(gs[base_row + 3, col_idx])
                    spacer_ax.set_visible(False)

                is_top_group = (row_idx == 0)
                is_bottom_group = (row_idx == n_skews - 1)
                is_leftmost_column = (col_idx == 0)

                self._plot_distribution(ax1, target_positions, temp_config, scale_val, skew_val,
                                      is_top_row=is_top_group, is_leftmost_column=is_leftmost_column)
                self._plot_standard_optimization(ax2, standard_trajectories, temp_config,
                                               is_bottom_row=False, is_leftmost_column=is_leftmost_column)
                self._plot_gaussian_optimization(ax3, gaussian_pos, gaussian_height, temp_config,
                                               is_bottom_row=is_bottom_group, is_leftmost_column=is_leftmost_column)

        plt.tight_layout()
        filename = 'bce_loss_grid_experiments.png'
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')

        return filename

    def _plot_distribution(self, ax, target_positions, config, scale_val, skew_val, is_top_row=True, is_leftmost_column=True):
        """Plot target distribution and path"""
        x_kde = np.linspace(config.X_MIN, config.X_MAX, 200)

        y_positions = np.linspace(0, 1.0, len(target_positions))
        target_colors = plt.cm.get_cmap('Greens')(np.linspace(0.3, 1.0, len(target_positions)))

        for i, (x_pos, y_pos, color) in enumerate(zip(target_positions, y_positions, target_colors)):
            ax.plot([x_pos], [y_pos], 'o', color=color, markersize=2, alpha=0.7,
                   markeredgewidth=0, zorder=3)

        try:
            kde = gaussian_kde(target_positions)
            kde_values = kde(x_kde)
            kde_scaled = kde_values / np.max(kde_values)
            ax.fill_between(x_kde, 0, kde_scaled, alpha=0.6, color='lightgray',
                          label='Target Distribution', zorder=1)
            ax.plot(x_kde, kde_scaled, color='black', linewidth=1.0, alpha=0.8, zorder=2)
        except:
            pass

        sm1 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Greens'),
                                   norm=plt.Normalize(vmin=0, vmax=len(target_positions)-1))
        sm1.set_array([])
        cax1 = inset_axes(ax, width="25%", height="4%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.08, 1, 1), bbox_transform=ax.transAxes)
        cbar1 = plt.colorbar(sm1, cax=cax1, orientation='horizontal', label='Step')
        cbar1.ax.tick_params(labelsize=6)
        cbar1.set_label('Step', fontsize=5)

        ax.plot([], [], 'o', color='darkgreen', markersize=3, alpha=0.7,
                markeredgewidth=0, label='Target Path')

        ax.tick_params(labelbottom=False)

        if is_leftmost_column:
            ax.set_ylabel('Density', fontsize=8)

        ax.set_title(f'Scale={scale_val}, Skew={skew_val}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(config.X_MIN, config.X_MAX)
        ax.tick_params(labelsize=7)
        ax.legend(loc='lower left', fontsize=6)

    def _plot_standard_optimization(self, ax, standard_trajectories, config, is_bottom_row=False, is_leftmost_column=True):
        """Plot standard optimization results"""
        steps_to_show = np.linspace(0, len(standard_trajectories[0]['heights']) - 1,
                                   min(7, config.NUM_DISPLAY_LINES), dtype=int)
        colors = plt.cm.get_cmap('Blues')(np.linspace(0.3, 1.0, len(steps_to_show)))

        for i, (step_idx, color) in enumerate(zip(steps_to_show, colors)):
            curve_times = [traj['time'] for traj in standard_trajectories]
            curve_heights = [traj['heights'][step_idx] for traj in standard_trajectories]

            sorted_data = sorted(zip(curve_times, curve_heights))
            sorted_times, sorted_heights = zip(*sorted_data)

            ax.plot(sorted_times, sorted_heights, 'o-', color=color,
                   markersize=1.5, linewidth=0.6, alpha=0.8)

        sm2 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Blues'),
                                  norm=plt.Normalize(vmin=0, vmax=max(steps_to_show)))
        sm2.set_array([])
        cax2 = inset_axes(ax, width="25%", height="4%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.08, 1, 1), bbox_transform=ax.transAxes)
        cbar2 = plt.colorbar(sm2, cax=cax2, orientation='horizontal', label='Step')
        cbar2.ax.tick_params(labelsize=6)
        cbar2.set_label('Step', fontsize=5)

        ax.plot([], [], 'o-', color='steelblue', markersize=2, linewidth=0.8, alpha=0.8,
                label='Point-wise BCE')

        ax.tick_params(labelbottom=False)

        if is_leftmost_column:
            ax.set_ylabel('Amplitude', fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(config.X_MIN, config.X_MAX)
        ax.tick_params(labelsize=7)
        ax.legend(loc='upper left', fontsize=6)

    def _plot_gaussian_optimization(self, ax, gaussian_pos, gaussian_height, config, is_bottom_row=False, is_leftmost_column=True):
        """Plot Gaussian-constrained optimization results"""
        truncate_range = 60 * config.SIGMA
        template_length = 2 * truncate_range + 1
        template_center = template_length // 2
        template_curve = self._gaussian_curve(template_length, template_center, config.SIGMA, 1.0)

        display_time = (np.arange(config.SIGNAL_LENGTH) - config.SIGNAL_CENTER) / config.SAMPLING_RATE
        mask = (display_time >= config.X_MIN) & (display_time <= config.X_MAX)

        steps_to_show = np.linspace(0, len(gaussian_pos) - 1, min(7, config.NUM_DISPLAY_LINES), dtype=int)
        colors = plt.cm.get_cmap('Reds')(np.linspace(0.3, 1.0, len(steps_to_show)))

        for idx, (step_idx, color) in enumerate(zip(steps_to_show, colors)):
            pos = gaussian_pos[step_idx]
            height = gaussian_height[step_idx]

            pred_curve = np.zeros(config.SIGNAL_LENGTH)
            pos_samples = pos * config.SAMPLING_RATE + config.SIGNAL_CENTER

            start_idx = int(pos_samples - truncate_range)
            end_idx = int(pos_samples + truncate_range + 1)

            template_start = max(0, -start_idx)
            template_end = min(template_length, config.SIGNAL_LENGTH - start_idx)
            pred_start = max(0, start_idx)
            pred_end = min(config.SIGNAL_LENGTH, end_idx)

            if pred_start < pred_end and template_start < template_end:
                template_slice = template_curve[template_start:template_end] * height
                pred_curve[pred_start:pred_end] = template_slice

            ax.plot(display_time[mask], pred_curve[mask], color=color, linewidth=0.8, alpha=0.8)

        sm3 = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('Reds'),
                                   norm=plt.Normalize(vmin=0, vmax=max(steps_to_show)))
        sm3.set_array([])
        cax3 = inset_axes(ax, width="25%", height="4%", loc='upper right',
                          bbox_to_anchor=(-0.02, -0.08, 1, 1), bbox_transform=ax.transAxes)
        cbar3 = plt.colorbar(sm3, cax=cax3, orientation='horizontal', label='Step')
        cbar3.ax.tick_params(labelsize=6)
        cbar3.set_label('Step', fontsize=5)

        ax.plot([], [], '-', color='crimson', linewidth=1.2, alpha=0.8,
                label='Gaussian-Constrained BCE')

        ax.set_xlabel('Time (s)', fontsize=8)
        ax.tick_params(labelbottom=True)

        if is_leftmost_column:
            ax.set_ylabel('Amplitude', fontsize=8)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(config.X_MIN, config.X_MAX)
        ax.tick_params(labelsize=7)
        ax.legend(loc='upper left', fontsize=6)

    def _gaussian_curve(self, length, peak_pos, sigma, height=1.0):
        """Generate Gaussian curve for visualization"""
        x = np.arange(length)
        curve = np.exp(-0.5 * ((x - peak_pos) / sigma) ** 2)
        return (curve / np.max(curve)) * height


def main():
    """Main experiment runner - Three plot visualization"""
    config = Config()
    target_generator = SkewedTargetGenerator(config)
    standard_optimizer = StandardBCEOptimizer(config)
    gaussian_optimizer = GaussianBCEOptimizer(config)
    visualizer = BCEVisualizerThreePlots(config)

    targets, target_positions = target_generator.generate_skewed_targets(config.NUM_STEPS)
    standard_trajectories = standard_optimizer.optimize_to_targets(targets)
    gaussian_pos, gaussian_height = gaussian_optimizer.optimize_to_targets(targets)

    filename = visualizer.create_three_plot_visualization(
        targets, target_positions, standard_trajectories, gaussian_pos, gaussian_height)

    return filename


def main_grid_experiment():
    """Main experiment runner - Grid visualization with multiple parameter combinations"""
    scale_values = [0.1, 0.2, 0.3, 0.5]
    skew_values = [0, -10]

    config = Config()
    grid_visualizer = BCEVisualizerGridPlots(config)

    filename = grid_visualizer.create_grid_visualization(scale_values, skew_values)

    return filename


if __name__ == "__main__":
    grid_filename = main_grid_experiment()

