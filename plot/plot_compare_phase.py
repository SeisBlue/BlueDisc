import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd

data = sbd.InstanceCounts()

source_counts = data.metadata['source_id'].value_counts()
selected_source = source_counts.index[20]
event_metadata = data.metadata[data.metadata['source_id'] == selected_source].copy()

p_arrival_col = 'trace_p_arrival_sample' if 'trace_p_arrival_sample' in event_metadata.columns else 'trace_P_arrival_sample'
s_arrival_col = 'trace_s_arrival_sample' if 'trace_s_arrival_sample' in event_metadata.columns else 'trace_S_arrival_sample'

if 'source_distance_km' in event_metadata.columns:
    distance_col = 'source_distance_km'
elif 'path_ep_distance_km' in event_metadata.columns:
    distance_col = 'path_ep_distance_km'
else:
    event_metadata['distance'] = range(len(event_metadata))
    distance_col = 'distance'

event_metadata = event_metadata.sort_values(distance_col)
event_metadata = event_metadata[
    event_metadata[p_arrival_col].notna() &
    event_metadata[s_arrival_col].notna() &
    (event_metadata[distance_col] <= 100)
]

grid_spacing = 5
grid_points = np.arange(
    np.floor(event_metadata[distance_col].min() / grid_spacing) * grid_spacing,
    np.ceil(event_metadata[distance_col].max() / grid_spacing) * grid_spacing + grid_spacing,
    grid_spacing
)

selected_indices = []
for grid_point in grid_points:
    closest_idx = np.abs(event_metadata[distance_col] - grid_point).idxmin()
    if closest_idx not in selected_indices:
        selected_indices.append(closest_idx)

event_metadata = event_metadata.loc[selected_indices]

waveforms = []
distances = []
p_arrivals = []
s_arrivals = []

sampling_rate = event_metadata['trace_sampling_rate_hz'].iloc[0] if 'trace_sampling_rate_hz' in event_metadata.columns else 100

for idx, row in event_metadata.iterrows():
    waveforms.append(data.get_waveforms(idx)[0, :])
    distances.append(row[distance_col])
    p_arrivals.append(row[p_arrival_col] if not pd.isna(row[p_arrival_col]) else None)
    s_arrivals.append(row[s_arrival_col] if not pd.isna(row[s_arrival_col]) else None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

scale_factor = 0.8
mean_distance_diff = np.mean(np.diff(distances)) if len(distances) > 1 and np.mean(np.diff(distances)) != 0 else 1

for waveform, distance, p_arrival, s_arrival in zip(waveforms, distances, p_arrivals, s_arrivals):
    waveform_normalized = waveform / (np.abs(waveform).max() + 1e-10)
    waveform_scaled = waveform_normalized * mean_distance_diff * scale_factor
    time = np.arange(len(waveform)) / sampling_rate

    if p_arrival is not None:
        p_arrival_time = p_arrival / sampling_rate
        time = time - p_arrival_time

    ax1.plot(time, waveform_scaled + distance, 'k', linewidth=0.3, alpha=0.8)

    if s_arrival is not None and p_arrival is not None:
        s_arrival_time = s_arrival / sampling_rate
        line_half_length = mean_distance_diff * scale_factor * 0.5
        ax1.plot([s_arrival_time - p_arrival_time, s_arrival_time - p_arrival_time],
                [distance - line_half_length, distance + line_half_length],
                '#FFB6C1', linewidth=1, alpha=1)

ax1.axvline(x=0, color='#AEC6CF', linestyle='-', linewidth=0.5, alpha=0.7, label='P Label')
ax1.set_xlabel('Time relative to P-wave arrival (s)', fontsize=12)
ax1.set_ylabel(f'{distance_col.replace("_", " ").title()} (km)', fontsize=12)
ax1.set_title(f'P-wave Aligned\nSource ID: {selected_source}', fontsize=14, fontweight='bold')
ax1.set_xlim(-10, 10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')

for waveform, distance, p_arrival, s_arrival in zip(waveforms, distances, p_arrivals, s_arrivals):
    waveform_normalized = waveform / (np.abs(waveform).max() + 1e-10)
    waveform_scaled = waveform_normalized * mean_distance_diff * scale_factor
    time = np.arange(len(waveform)) / sampling_rate

    if s_arrival is not None:
        s_arrival_time = s_arrival / sampling_rate
        time = time - s_arrival_time

    ax2.plot(time, waveform_scaled + distance, 'k', linewidth=0.3, alpha=0.8)

    if p_arrival is not None and s_arrival is not None:
        p_arrival_time = p_arrival / sampling_rate
        line_half_length = mean_distance_diff * scale_factor * 0.5
        ax2.plot([p_arrival_time - s_arrival_time, p_arrival_time - s_arrival_time],
                [distance - line_half_length, distance + line_half_length],
                '#AEC6CF', linewidth=1, alpha=1)

ax2.axvline(x=0, color='#FFB6C1', linestyle='-', linewidth=0.5, alpha=0.7, label='S Label')
ax2.set_xlabel('Time relative to S-wave arrival (s)', fontsize=12)
ax2.set_ylabel(f'{distance_col.replace("_", " ").title()} (km)', fontsize=12)
ax2.set_title(f'S-wave Aligned\nSource ID: {selected_source}', fontsize=14, fontweight='bold')
ax2.set_xlim(-10, 10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('event_waveforms_comparison_p_s_aligned.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: event_waveforms_comparison_p_s_aligned.png")

