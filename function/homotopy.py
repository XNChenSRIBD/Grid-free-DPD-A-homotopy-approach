"""Gaussian continuation with an adaptive predictor-corrector step in alpha."""

import numpy as np


def homotopy(covariance, sensors, frequencies, alpha0, tau=0.001, initial=None):
    """Return the continuous position estimate inside the sensor polygon.

    Sensors are ordered around the convex polygon. Alpha decreases from alpha0
    to zero; each corrector maximizes the smoothed DPD objective using projected
    Barzilai-Borwein steps and Armijo backtracking. Tau bounds the correction
    relative to the polygon diameter, rather than relative to the unknown truth.
    """
    c = 299792458.0
    bandwidth = 1e7
    pg_tolerance = 1e-4
    max_updates = 2000
    max_backtracks = 30
    epsilon = np.finfo(float).eps
    edges = np.roll(sensors, -1, axis=0) - sensors
    edge_lengths_squared = np.sum(edges * edges, axis=1)
    edge_lengths = np.sqrt(edge_lengths_squared)
    orientation = np.sign(np.sum(
        sensors[:, 0] * edges[:, 1] - sensors[:, 1] * edges[:, 0]
    ))
    differences = sensors[:, None, :] - sensors[None, :, :]
    diameter_squared = float(np.max(np.sum(differences * differences, axis=2)))
    diameter = np.sqrt(diameter_squared)
    omega = 2.0 * np.pi * frequencies
    spectral_coordinate = (2.0 * frequencies / bandwidth) ** 2
    trace = np.trace(covariance, axis1=1, axis2=2).real
    difference_step = np.cbrt(epsilon) * diameter
    inward_normals = orientation * np.column_stack(
        (-edges[:, 1], edges[:, 0])
    ) / edge_lengths[:, None]

    def project(point):
        relative = point - sensors
        cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
        if np.all(orientation * cross >= 0.0):
            return point.copy()
        fractions = np.clip(
            np.sum(relative * edges, axis=1) / edge_lengths_squared, 0.0, 1.0
        )
        nearest = sensors + fractions[:, None] * edges
        return nearest[np.argmin(np.sum((nearest - point) ** 2, axis=1))].copy()

    def correct(alpha, start):
        weights = np.exp(-alpha * spectral_coordinate)
        weighted_covariance = weights[:, None, None] * covariance
        normalization = float(np.dot(weights, trace))

        def matrix_terms(point):
            displacement = point - sensors
            distances = np.linalg.norm(displacement, axis=1)
            phases = np.exp(-1j * omega[:, None] * distances[None, :] / c)
            terms = weighted_covariance * phases.conj()[:, :, None] * phases[:, None, :]
            matrix = terms.sum(axis=0)
            return displacement, distances, terms, 0.5 * (matrix + matrix.conj().T)

        def evaluate(point, gradient=False):
            displacement, distances, terms, matrix = matrix_terms(point)
            if not gradient:
                return float(np.linalg.eigvalsh(matrix)[-1] / normalization)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            value = float(eigenvalues[-1] / normalization)
            if np.any(distances == 0.0) or eigenvalues[-1] == eigenvalues[-2]:
                return value, None
            delay_gradient = displacement / (c * distances[:, None])
            frequency_sum = np.sum((1j * omega)[:, None, None] * terms, axis=0)
            matrix_gradient = (
                delay_gradient.T[:, :, None] - delay_gradient.T[:, None, :]
            ) * frequency_sum[None, :, :]
            eigenvector = eigenvectors[:, -1]
            # For a simple largest eigenvalue: d lambda = u^H (d M) u.
            derivative = np.einsum(
                "m,amn,n->a", eigenvector.conj(), matrix_gradient, eigenvector
            ).real / normalization
            return value, derivative

        def projected_residual(point, gradient):
            return point - project(point + diameter_squared * gradient)

        def position_precision(point, gradient):
            # Linearize the projected stationarity equation R(p)=0. This keeps
            # a small gradient in a flat landscape from ending correction early.
            jacobian = np.empty((2, 2))
            hessian = np.empty((2, 2))
            for axis in range(2):
                offset = np.zeros(2)
                offset[axis] = difference_step
                plus, minus = point + offset, point - offset
                _, gradient_plus = evaluate(plus, True)
                _, gradient_minus = evaluate(minus, True)
                if (gradient_plus is None or gradient_minus is None
                        or not np.all(np.isfinite(gradient_plus))
                        or not np.all(np.isfinite(gradient_minus))):
                    return False, None
                jacobian[:, axis] = (
                    projected_residual(plus, gradient_plus)
                    - projected_residual(minus, gradient_minus)
                ) / (2.0 * difference_step)
                hessian[:, axis] = (gradient_plus - gradient_minus) / (2.0 * difference_step)
            hessian = 0.5 * (hessian + hessian.T)
            if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(hessian)):
                return False, None
            if np.linalg.cond(jacobian) >= 1.0 / np.sqrt(epsilon):
                return False, hessian
            delta = np.linalg.solve(jacobian, -projected_residual(point, gradient))
            prediction = point + delta
            feasibility_error = np.linalg.norm(project(prediction) - prediction)
            feasibility_tolerance = 64.0 * epsilon * max(
                1.0, diameter, np.linalg.norm(point), np.linalg.norm(prediction)
            )
            precise = (np.linalg.norm(delta) / diameter <= tau / 10.0
                       and feasibility_error <= feasibility_tolerance)
            return precise, hessian

        def stable_maximum(point, gradient, value, hessian):
            # At an interior maximum all curvatures must be negative. On an
            # edge, only tangential curvature matters for an outward gradient.
            position_scale = max(
                1.0, diameter, np.linalg.norm(point), np.max(np.linalg.norm(sensors, axis=1))
            )
            boundary_roundoff = 64.0 * epsilon * position_scale
            gradient_scale = max(
                np.linalg.norm(gradient), np.linalg.norm(hessian, ord=2) * diameter,
                abs(value) / diameter, 1.0 / diameter,
            )
            gradient_roundoff = np.sqrt(epsilon) * gradient_scale
            curvature_roundoff = np.sqrt(epsilon) * max(
                np.linalg.norm(hessian, ord=2), gradient_scale / diameter
            )
            relative = point - sensors
            signed_distance = orientation * (
                edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
            ) / edge_lengths
            active = np.flatnonzero(signed_distance <= boundary_roundoff)
            largest_curvature = np.linalg.eigvalsh(hessian)[-1]
            if np.min(signed_distance) < -boundary_roundoff:
                return False
            if len(active) == 0:
                return largest_curvature < -curvature_roundoff
            if len(active) == 1:
                edge = active[0]
                tangent = edges[edge] / edge_lengths[edge]
                inward_gradient = gradient @ inward_normals[edge]
                if (inward_gradient > gradient_roundoff
                        or tangent @ hessian @ tangent >= -curvature_roundoff):
                    return False
                if inward_gradient >= -gradient_roundoff:
                    return largest_curvature < -curvature_roundoff
                return True
            return False

        def stationary_maximum(point, value, hessian):
            # At machine-level stationarity, rule out a saddle by probing the
            # feasible Hessian directions and any nearby polygon edge.
            _, eigenvectors = np.linalg.eigh(hessian)
            directions = [eigenvectors[:, 0], eigenvectors[:, 1]]
            relative = point - sensors
            distance = np.abs(
                edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
            ) / edge_lengths
            for edge in np.flatnonzero(distance <= difference_step):
                directions.append(edges[edge] / edge_lengths[edge])
            roundoff = 64.0 * epsilon * max(1.0, abs(value))
            for direction in directions:
                for sign in (-1.0, 1.0):
                    probe_value = evaluate(project(point + sign * difference_step * direction))
                    if not np.isfinite(probe_value) or probe_value > value + roundoff:
                        return False
            return True

        def vertex_direction(point, vertex):
            # Distance is nonsmooth at a sensor. Its one-sided derivative is
            # b^T v + kappa for unit directions v in the feasible vertex cone.
            displacement, distances, terms, matrix = matrix_terms(point)
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            value = float(eigenvalues[-1] / normalization)
            if eigenvalues[-1] == eigenvalues[-2]:
                return value, None, None
            eigenvector = eigenvectors[:, -1]
            frequency_sum = np.sum((1j * omega)[:, None, None] * terms, axis=0)
            q = eigenvector.conj()[:, None] * frequency_sum * eigenvector[None, :]
            sensitivity = (q.sum(axis=1) - q.sum(axis=0)).real / (normalization * c)
            regular = np.arange(len(sensors)) != vertex
            linear = np.sum(
                sensitivity[regular, None] * displacement[regular] / distances[regular, None], axis=0
            )
            directions = [sensors[(vertex - 1) % len(sensors)] - point,
                          sensors[(vertex + 1) % len(sensors)] - point]
            directions = [v / np.linalg.norm(v) for v in directions]
            if np.linalg.norm(linear) > 0.0:
                direction = linear / np.linalg.norm(linear)
                active = [(vertex - 1) % len(sensors), vertex]
                if np.all(inward_normals[active] @ direction >= -64.0 * epsilon):
                    directions.append(direction)
            derivatives = np.asarray(directions) @ linear + sensitivity[vertex]
            index = np.argmax(derivatives)
            return value, float(derivatives[index]), directions[index]

        point = start.copy()
        previous_point = previous_gradient = None
        last_step = diameter_squared
        for update in range(max_updates + 1):
            vertices = np.flatnonzero(np.all(point == sensors, axis=1))
            if len(vertices):
                value, derivative, direction = vertex_direction(point, vertices[0])
                if derivative is None or not np.isfinite(value) or not np.isfinite(derivative):
                    return point, False
                if derivative <= 0.0:
                    return point, True
                if update == max_updates:
                    return point, False
                slack = np.einsum("ij,ij->i", inward_normals, point - sensors)
                slope = inward_normals @ direction
                exiting = slope < -64.0 * epsilon
                feasible_length = np.min(np.maximum(slack[exiting], 0.0) / -slope[exiting])
                ray_length = min(diameter_squared * derivative, feasible_length)
                for backtrack in range(max_backtracks):
                    length = ray_length * 0.5 ** backtrack
                    candidate = project(point + length * direction)
                    candidate_value = evaluate(candidate)
                    if np.isfinite(candidate_value) and candidate_value >= value + 1e-4 * derivative * length:
                        if np.array_equal(candidate, point):
                            return point, False
                        point = candidate
                        previous_point = previous_gradient = None
                        last_step = diameter_squared
                        break
                else:
                    return point, False
                continue

            value, gradient = evaluate(point, True)
            if gradient is None or not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return point, False
            residual = np.linalg.norm(projected_residual(point, gradient)) / diameter
            if not np.isfinite(residual):
                return point, False
            if residual <= pg_tolerance:
                precise, hessian = position_precision(point, gradient)
                if precise:
                    stable = stable_maximum(point, gradient, value, hessian)
                    if not stable and residual <= 64.0 * epsilon:
                        return point, False
                    if stable:
                        if residual <= 64.0 * epsilon and not stationary_maximum(point, value, hessian):
                            return point, False
                        return point, True
            if update == max_updates:
                return point, False

            proposed_step = last_step
            if previous_point is not None:
                displacement = point - previous_point
                # Maximization changes the sign of the BB curvature vector.
                gradient_difference = previous_gradient - gradient
                denominator = np.dot(displacement, gradient_difference)
                if denominator > 0.0:
                    bb_step = np.dot(displacement, displacement) / denominator
                    if np.isfinite(bb_step) and bb_step > 0.0:
                        proposed_step = bb_step
            for backtrack in range(max_backtracks):
                step = proposed_step * 0.5 ** backtrack
                candidate = point + step * gradient
                if not np.all(np.isfinite(candidate)):
                    continue
                candidate = project(candidate)
                candidate_value = evaluate(candidate)
                if (np.isfinite(candidate_value)
                        and candidate_value >= value + 1e-4 * np.dot(gradient, candidate - point)):
                    if np.array_equal(candidate, point):
                        return point, False
                    previous_point, previous_gradient = point.copy(), gradient.copy()
                    point, last_step = candidate, step
                    break
            else:
                return point, False

    initial = np.zeros(2) if initial is None else np.asarray(initial, dtype=float)
    point, success = correct(alpha0, initial)
    if not success:
        return point
    alpha = step = float(alpha0)
    previous_point = previous_alpha = None
    rejected_pairs = set()
    while alpha > 0.0:
        step = min(step, alpha)
        trial_alpha = 0.0 if step == alpha else alpha - step
        effective_step = alpha - trial_alpha
        if (not 0.0 <= trial_alpha < alpha or effective_step <= 0.0
                or (alpha, trial_alpha) in rejected_pairs):
            return point
        predictor = point.copy()
        if previous_point is not None:
            predictor += effective_step / (previous_alpha - alpha) * (point - previous_point)
        corrected, success = correct(trial_alpha, project(predictor))
        # The correction is measured from the unprojected secant predictor.
        error = np.linalg.norm(corrected - predictor) / diameter
        if success and np.isfinite(error) and error <= tau:
            previous_point, previous_alpha = point.copy(), alpha
            point, alpha = corrected, trial_alpha
            step = min(2.0 * effective_step, alpha)
        else:
            rejected_pairs.add((alpha, trial_alpha))
            step = effective_step / 2.0
    return point
