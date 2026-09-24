"""Display the original ML landscape and the four position estimates."""

import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
from matplotlib.patches import Rectangle
import numpy as np

from .grid_methods import ml_spectrum


def plot_landscape(covariance, sensors, frequencies, truth, estimates,
                   runtimes_ms, grid_axis, half_width, output_path, snr_db):
    """Dense maps are evaluated here only after all estimates have been computed."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "mathtext.fontset": "stix",
    })
    axis = np.linspace(-half_width, half_width, 301)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    landscape = ml_spectrum(covariance, points, sensors, frequencies).reshape(x.shape)
    limits = {"vmin": landscape.min(), "vmax": landscape.max()}
    fig, ax = plt.subplots(figsize=(8.6, 6.9))
    fig.subplots_adjust(left=0.10, right=0.87, bottom=0.09, top=0.81)
    heatmap = ax.pcolormesh(x, y, landscape, shading="auto", cmap="viridis", **limits)
    ax.contour(x, y, landscape, levels=np.linspace(landscape.min(), landscape.max(), 13)[1:-1],
               colors="white", linewidths=0.35, alpha=0.75)
    fig.colorbar(heatmap, ax=ax, fraction=0.047, pad=0.035,
                 label="Normalized ML spectrum")

    styles = {
        "Vanilla": ("o", "#0072B2", 12),
        "Contextual": ("^", "#E69F00", 10),
        "LRMC": ("s", "#009E73", 8),
        "Homotopy": ("D", "#D55E00", 6),
    }

    def mark_positions(panel):
        panel.plot(*truth, marker="*", markersize=13, color="black",
                   markeredgecolor="white", markeredgewidth=0.8, linestyle="none",
                   label=r"Ground-truth location $\boldsymbol{p}_{\natural}$", zorder=4)
        for name, position in estimates.items():
            marker, color, size = styles[name]
            panel.plot(*position, marker=marker, markersize=size, color=color,
                       markerfacecolor="none", markeredgewidth=1.6, linestyle="none",
                       label=f"{name} ({runtimes_ms[name]:.2f} ms)", zorder=5,
                       path_effects=[effects.Stroke(linewidth=2.4, foreground="white"),
                                     effects.Normal()])

    mark_positions(ax)
    ax.plot(sensors[:, 0], sensors[:, 1], marker="v", markersize=8,
            color="white", markeredgecolor="black", linestyle="none",
            label="Receivers", zorder=3, clip_on=False)
    ax.set(xlim=(-half_width, half_width), ylim=(-half_width, half_width),
           xlabel=r"$x$ (m)", ylabel=r"$y$ (m)", aspect="equal")
    scene_size = (f"{2 * half_width:g} m" if half_width < 500
                  else f"{2 * half_width / 1000:g} km")
    ax.set_title(f"{scene_size} × {scene_size}, SNR = {snr_db:g} dB", pad=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.49, 0.995), frameon=True, edgecolor="black",
               fontsize=9.5, columnspacing=1.2, handletextpad=0.5)

    # A local view resolves nearby estimates without moving any plotted point.
    nearby = np.array([truth, estimates["Vanilla"], estimates["Contextual"],
                       estimates["Homotopy"]])
    grid_spacing = grid_axis[1] - grid_axis[0]
    radius = max(1.6 * grid_spacing, 1.3 * np.max(np.abs(nearby - truth)))
    zoom_x = np.linspace(truth[0] - radius, truth[0] + radius, 151)
    zoom_y = np.linspace(truth[1] - radius, truth[1] + radius, 151)
    zx, zy = np.meshgrid(zoom_x, zoom_y)
    zoom_points = np.column_stack((zx.ravel(), zy.ravel()))
    zoom_map = ml_spectrum(covariance, zoom_points, sensors, frequencies).reshape(zx.shape)
    # A white surround keeps the title, coordinates, and units off the heatmap.
    ax.add_patch(Rectangle((0.49, 0.49), 0.46, 0.48, transform=ax.transAxes,
                          facecolor="white", edgecolor="0.3", linewidth=0.7, zorder=3))
    inset = ax.inset_axes([0.58, 0.60, 0.34, 0.33], zorder=6)
    inset.pcolormesh(zx, zy, zoom_map, shading="auto", cmap="viridis", **limits)
    inset.contour(zx, zy, zoom_map, levels=np.linspace(zoom_map.min(), zoom_map.max(), 7)[1:-1],
                  colors="white", linewidths=0.4, alpha=0.8)
    # Ticks and dashed lines are the actual 101 x 101 search-grid coordinates.
    grid_x = grid_axis[(grid_axis >= zoom_x[0]) & (grid_axis <= zoom_x[-1])]
    grid_y = grid_axis[(grid_axis >= zoom_y[0]) & (grid_axis <= zoom_y[-1])]
    inset.set_xticks(grid_x)
    inset.set_yticks(grid_y)
    inset.grid(True, color="0.2", linestyle="--", linewidth=0.65, alpha=0.8)
    gx, gy = np.meshgrid(grid_x, grid_y)
    inset.scatter(gx, gy, s=9, c="0.2", edgecolors="white", linewidths=0.35, zorder=3)
    mark_positions(inset)
    inset.set(xlim=(zoom_x[0], zoom_x[-1]), ylim=(zoom_y[0], zoom_y[-1]), aspect="equal")
    inset.set_title(f"Local view ({grid_spacing:g} m grid)", fontsize=9, pad=4)
    inset.set_xlabel(r"$x$ (m)", fontsize=8, labelpad=2)
    inset.set_ylabel(r"$y$ (m)", fontsize=8, labelpad=2)
    inset.tick_params(labelsize=8, direction="in", length=3, pad=3)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
