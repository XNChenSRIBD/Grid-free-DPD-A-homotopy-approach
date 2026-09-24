"""Four-method localization demo in a 200 m x 200 m scene."""

from pathlib import Path
import numpy as np

from function.observation import generate_observation
from function.grid_methods import vanilla, contextual, lrmc
from function.homotopy import homotopy
from function.plot_landscape import plot_landscape

snr_db = -10
seed = 20260924
sensors = np.array([[-95., 31.], [-59., -81.], [59., -81.], [95., 31.], [0., 100.]])
truth = np.array([-27.35, -16.42])
half_width = 100.
axis = np.linspace(-half_width, half_width, 101)  # 2 m grid spacing
xx, yy = np.meshgrid(axis, axis)
points = np.column_stack((xx.ravel(), yy.ravel()))

covariance, frequencies = generate_observation(sensors, truth, cp_length=16,
                                               snr_db=snr_db, seed=seed)
estimates = {
    "Vanilla": vanilla(covariance, points, sensors, frequencies),
    "Contextual": contextual(covariance, points, sensors, frequencies),
    "LRMC": lrmc(covariance, points, xx.shape, sensors, frequencies, seed=seed),
    "Homotopy": homotopy(covariance, sensors, frequencies, alpha0=24.),
}
for name, point in estimates.items():
    print(f"{name:10s} ({point[0]:9.4f}, {point[1]:9.4f}) m; error = {np.linalg.norm(point - truth):.4f} m")

output = Path(__file__).resolve().parent
np.savez(output / "results_200m.npz", sensors=sensors, truth=truth, covariance=covariance,
         frequencies=frequencies, snr_db=snr_db, seed=seed, **estimates)
plot_landscape(covariance, sensors, frequencies, truth, estimates, half_width,
               output / "landscape_200m.png")
