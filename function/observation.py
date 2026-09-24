"""A shared CP-OFDM/QPSK observation for the four localization methods."""

import numpy as np


def generate_observation(sensors, truth, cp_length, snr_db=-10, seed=20260924):
    """Return sample covariance (frequency, station, station) and frequencies.

    SNR is the received time-domain signal-to-noise power ratio at station 1.
    All stations use the same noise variance and a common receive FFT window.
    The channel is constant across the 128 received OFDM symbols.
    """
    rng = np.random.default_rng(seed)
    sensors = np.asarray(sensors, dtype=float)
    truth = np.asarray(truth, dtype=float)
    fft_size = 128
    snapshots = 128
    sample_rate = 12.8e6
    c0 = 299792458.0
    active_bins = np.arange(-50, 51)
    fft_bins = active_bins % fft_size
    frequencies = active_bins * sample_rate / fft_size
    symbol_length = fft_size + cp_length

    # Power decays as distance^-2; log-amplitude shadowing has variance 0.003.
    distances = np.linalg.norm(truth - sensors, axis=1)
    shadow = np.exp(rng.normal(0.0, np.sqrt(0.003), len(sensors)))
    phase = rng.uniform(0.0, 2.0 * np.pi, len(sensors))
    channel = distances**-1 * shadow * np.exp(1j * phase)
    delays_samples = distances / c0 * sample_rate

    # One preceding symbol supplies the history of the first received CP.
    total_symbols = snapshots + 1
    real = 2 * rng.integers(0, 2, (total_symbols, len(active_bins))) - 1
    imag = 2 * rng.integers(0, 2, (total_symbols, len(active_bins))) - 1
    transmitted = np.zeros((total_symbols, fft_size), dtype=complex)
    transmitted[:, fft_bins] = (real + 1j * imag) / np.sqrt(2.0)

    # Evaluate s(t-tau) within each symbol with its N-point Fourier series.
    # Then add the CP and delay the concatenated stream by integer samples.
    signed_bins = np.fft.fftfreq(fft_size) * fft_size
    clean = np.empty((snapshots, symbol_length, len(sensors)), dtype=complex)
    for station, delay in enumerate(delays_samples):
        integer_delay = int(np.ceil(delay))
        fractional_advance = integer_delay - delay
        fractional_phase = np.exp(
            2j * np.pi * signed_bins * fractional_advance / fft_size
        )
        symbol_time = np.fft.ifft(
            transmitted * fractional_phase, axis=1, norm="ortho"
        )
        stream = np.concatenate(
            (symbol_time[:, -cp_length:], symbol_time), axis=1
        ).ravel()
        first = symbol_length - integer_delay
        last = first + snapshots * symbol_length
        clean[:, :, station] = (
            stream[first:last].reshape(snapshots, symbol_length) * channel[station]
        )

    signal_power = np.mean(np.abs(clean[:, cp_length:, 0]) ** 2)
    noise_variance = signal_power / 10.0 ** (snr_db / 10.0)
    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    )
    # Remove the same CP samples at every station, then keep the active bins.
    received = (clean + noise)[:, cp_length:, :]
    received_fft = np.fft.fft(received, axis=1, norm="ortho")
    active_received = received_fft[:, fft_bins, :].transpose(1, 0, 2)
    covariance = np.einsum(
        "ksm,ksn->kmn", active_received, active_received.conj()
    ) / snapshots
    return covariance, frequencies
