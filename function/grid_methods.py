"""Static-grid Vanilla DPD, contextual MVDR, and ML-map completion."""

import numpy as np
from scipy.special import ndtri

C0 = 299792458.0


def ml_spectrum(covariance, points, sensors, frequencies):
    """Normalized largest eigenvalue of the phase-compensated covariance sum."""
    normalization = np.trace(covariance, axis1=1, axis2=2).real.sum()
    values = np.empty(len(points))
    # Small blocks also allow the same function to draw a dense heatmap.
    for start in range(0, len(points), 1024):
        block = points[start:start + 1024]
        delays = np.linalg.norm(block[:, None] - sensors[None], axis=2) / C0
        steering = np.exp(-2j * np.pi * frequencies[None, :, None] * delays[:, None])
        matrix = np.einsum("gkm,kmn,gkn->gmn", steering.conj(), covariance,
                           steering, optimize=True)
        matrix = (matrix + matrix.conj().swapaxes(-1, -2)) / 2
        values[start:start + len(block)] = np.linalg.eigvalsh(matrix)[:, -1] / normalization
    return values


def vanilla(covariance, points, sensors, frequencies):
    """Choose the grid point with the largest ML spectrum value."""
    values = ml_spectrum(covariance, points, sensors, frequencies)
    return points[np.argmax(values)].copy()


def project_amplitudes(direction, lower, upper):
    """Project amplitudes onto ratio bounds relative to receiver 1, then normalize."""
    other = direction[:, 1:]
    breaks = np.concatenate((other / upper, other / lower), axis=1)
    order = np.argsort(breaks, axis=1)
    knots = np.take_along_axis(breaks, order, axis=1)
    delta_a = np.take_along_axis(np.concatenate((-upper**2, lower**2), axis=1), order, axis=1)
    delta_c = np.take_along_axis(np.concatenate((-upper * other, lower * other), axis=1), order, axis=1)
    a0 = 1 + np.sum(upper**2, axis=1, keepdims=True)
    c0 = direction[:, :1] + np.sum(upper * other, axis=1, keepdims=True)
    a = np.concatenate((a0, a0 + np.cumsum(delta_a, axis=1)), axis=1)
    c = np.concatenate((c0, c0 + np.cumsum(delta_c, axis=1)), axis=1)
    interval = np.sum(a[:, :-1] * knots - c[:, :-1] < 0, axis=1)
    row = np.arange(len(direction))
    reference = (c[row, interval] / a[row, interval])[:, None]
    amplitude = np.concatenate((reference, np.clip(other, lower * reference, upper * reference)), axis=1)
    return amplitude / np.linalg.norm(amplitude, axis=1, keepdims=True)


def contextual(covariance, points, sensors, frequencies):
    """Contextual-Exact: constrained MVDR, gamma=2, 99% shadowing interval."""
    distances = np.linalg.norm(points[:, None] - sensors[None], axis=2)
    steering = np.exp(-2j * np.pi * frequencies[None, :, None] * distances[:, None] / C0)
    matrix = np.einsum("gkm,kmn,gkn->gmn", steering.conj(), np.linalg.inv(covariance),
                       steering, optimize=True)
    matrix = (matrix + matrix.conj().swapaxes(-1, -2)) / 2

    # At a receiver the normalized path-loss vector has the one-hot limit.
    regular = np.all(distances > 0, axis=1)
    local_matrix = matrix[regular]
    eigenvalues, eigenvectors = np.linalg.eigh(local_matrix)
    shift = np.eye(len(sensors))[None] - local_matrix / eigenvalues[:, -1, None, None]
    radii = distances[regular]
    ratios = radii[:, :1] / radii[:, 1:]
    interval = np.sqrt(2 * 0.003) * ndtri((1 + 0.99) / 2)
    lower, upper = np.exp(-interval) * ratios, np.exp(interval) * ratios
    amplitude = 1 / radii
    amplitude /= np.linalg.norm(amplitude, axis=1, keepdims=True)
    channel = amplitude * np.exp(1j * np.angle(eigenvectors[:, :, 0]))

    active = np.ones(len(channel), dtype=bool)
    for _ in range(500):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        direction = np.einsum("gmn,gn->gm", shift[indices], channel[indices])
        amplitude = project_amplitudes(np.abs(direction), lower[indices], upper[indices])
        updated = amplitude * np.exp(1j * np.angle(direction))
        active[indices] = np.linalg.norm(updated - channel[indices], axis=1) > 1e-7
        channel[indices] = updated

    costs = np.empty(len(points))
    costs[regular] = np.einsum("gm,gmn,gn->g", channel.conj(), local_matrix, channel).real
    for index in np.flatnonzero(~regular):
        receiver = np.argmin(distances[index])
        costs[index] = matrix[index, receiver, receiver].real
    # Maximizing the negative MVDR cost is equivalent to this minimum.
    return points[np.argmin(costs)].copy()


def lrmc(covariance, points, shape, sensors, frequencies, seed=20260924):
    """Sample 30% of the ML map and complete it with 50 SVT-ADMM iterations."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(points))[:int(np.ceil(0.30 * len(points)))]
    scores = ml_spectrum(covariance, points[indices], sensors, frequencies)
    sparse_map = np.zeros(shape)
    sparse_map.flat[indices] = np.abs(scores) / np.mean(np.abs(scores))
    observed = sparse_map != 0
    estimate = sparse_map.copy()
    dual = sparse_map.copy()
    rho = 0.05
    for _ in range(50):
        left, singular, right = np.linalg.svd(estimate + dual / rho, full_matrices=False)
        low_rank = (left * np.maximum(singular - 1 / rho, 0)) @ right
        estimate = low_rank - dual / rho
        estimate[observed] = sparse_map[observed]
        dual -= rho * (low_rank - estimate)
    return points[np.argmax(estimate)].copy()
