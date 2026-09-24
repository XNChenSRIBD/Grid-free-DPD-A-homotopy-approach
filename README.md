# Grid-free DPD: a homotopy approach

Two small Python demos compare Vanilla DPD, Contextual-Exact, ML-LRMC, and adaptive Homotopy on the same noisy observation.

```bash
pip install -r requirements.txt
python demo_200m.py
python demo_1km.py
```

Each script prints the four estimated positions, Euclidean errors, and runtimes, saves the observation, estimates, and runtimes to `results_*.npz`, and draws `landscape_*.png`. The heatmap is the **unsmoothed, normalized maximum-eigenvalue DPD objective**, shown with the paper's viridis colormap and white contours. It marks the ground-truth location $\boldsymbol{p}_{\natural}$, all four estimates, and the receivers. The local view includes coordinate ticks and the actual static search grid (2 m or 10 m spacing); its dense heatmap samples are used only for plotting.

Each method's legend entry reports its Euclidean localization error in meters and the wall-clock time of that single complete algorithm call in milliseconds, measured with `time.perf_counter`. Observation generation, shared grid construction, and plotting are excluded. Runtime depends on the machine and numerical libraries.

## Scene and algorithms

Both scripts fix SNR to **-15 dB** and use `numpy.random.default_rng` with the scene-specific seeds below. A single seeded draw fixes the QPSK symbols, lognormal shadowing, channel phases, and noise; every algorithm receives the same sample covariance. The seeds are chosen to illustrate successful Homotopy localization in both scenes. Each scene is one reproducible realization, not a Monte Carlo average.

| Parameter | 200 m × 200 m | 1 km × 1 km |
|---|---|---|
| Random seed | 20260927 | 20260924 |
| Search region (each axis) | [-100, 100] m | [-500, 500] m |
| Source (x, y) | (-27.35, -16.42) m | (-136.75, -82.10) m |
| Static grid | 101 × 101 (2 m spacing) | 101 × 101 (10 m spacing) |
| Cyclic prefix | 16 samples | 64 samples |
| Initial Homotopy smoothing α | 24 | 250 |

The receiver coordinates are listed directly in each scene script. Both scenes use 128 independent CP-OFDM snapshots, a 128-point FFT, 101 active subcarriers (-50 through 50), and a 12.8 MHz sample rate (10 MHz occupied frequency span). Propagation retains the cyclic prefix and common receiver FFT window. The path-loss exponent is 2 and log-amplitude shadowing variance is 0.003. SNR refers to the clean time-domain power at receiver 1 divided by the common noise variance.

- **Vanilla:** largest eigenvalue of the phase-compensated covariance sum, maximized on the static grid.
- **Contextual:** minimum constrained MVDR cost; known path-loss exponent 2, 99% amplitude-ratio interval, MM tolerance 1e-7 and at most 500 updates per candidate.
- **LRMC:** evaluate a fixed seeded 30% sample of the same ML grid, normalize its scores, and complete the map with 50 SVT-ADMM iterations (ρ = 0.05).
- **Homotopy:** Gaussian frequency weighting, adaptive smoothing-parameter continuation, and BB–Armijo projected-gradient correction; starts at (0, 0), uses τ = 0.001, and ends at the unsmoothed objective. Projection uses the receiver convex hull.

All shared numerical and plotting functions are in `function/`. No previous experiment files are needed.
