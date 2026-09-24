"""Four-method localization demo in a 200 m x 200 m scene."""

from pathlib import Path
import time
import numpy as np

from function.observation import generate_observation
from function.grid_methods import vanilla, contextual, lrmc
from function.homotopy import homotopy
from function.plot_landscape import plot_landscape

snr_db = -15
seed = 20260927
sensors = np.array([[-95., 31.], [-59., -81.], [59., -81.], [95., 31.], [0., 100.]])
truth = np.array([-27.35, -16.42])
half_width = 100.
axis = np.linspace(-half_width, half_width, 101)  # 2 m grid spacing
xx, yy = np.meshgrid(axis, axis)
points = np.column_stack((xx.ravel(), yy.ravel()))

covariance, frequencies = generate_observation(sensors, truth, cp_length=16,
                                               snr_db=snr_db, seed=seed)
estimates = {}
runtimes_ms = {}

start = time.perf_counter()
estimates["Vanilla"] = vanilla(covariance, points, sensors, frequencies)
runtimes_ms["Vanilla"] = (time.perf_counter() - start) * 1000

start = time.perf_counter()
estimates["Contextual"] = contextual(covariance, points, sensors, frequencies)
runtimes_ms["Contextual"] = (time.perf_counter() - start) * 1000

start = time.perf_counter()
estimates["LRMC"] = lrmc(covariance, points, xx.shape, sensors, frequencies, seed=seed)
runtimes_ms["LRMC"] = (time.perf_counter() - start) * 1000

start = time.perf_counter()
estimates["Homotopy"] = homotopy(covariance, sensors, frequencies, alpha0=24.)
runtimes_ms["Homotopy"] = (time.perf_counter() - start) * 1000

for name, point in estimates.items():
    print(f"{name:10s} ({point[0]:9.4f}, {point[1]:9.4f}) m; "
          f"error = {np.linalg.norm(point - truth):.4f} m; time = {runtimes_ms[name]:.2f} ms")

output = Path(__file__).resolve().parent
np.savez(output / "results_200m.npz", sensors=sensors, truth=truth, covariance=covariance,
         frequencies=frequencies, snr_db=snr_db, seed=seed, **estimates,
         **{f"runtime_{name}_ms": elapsed for name, elapsed in runtimes_ms.items()})
plot_landscape(covariance, sensors, frequencies, truth, estimates, runtimes_ms, axis, half_width,
               output / "landscape_200m.png", snr_db)
