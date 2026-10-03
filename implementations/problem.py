"""Synthetic beamforming problem shared by all implementations, tests and benchmarks.

Every implementation in this directory computes, for the arrays returned by `make_problem`,

    B(x) = sum_w s^H A s          method "bartlett" (A = K), "crosscorrelation" (A = K without its diagonal)
    B(x) = mean_w 1 / (s^H A s)   method "mvdr" (A = inverse of K), "music" (A = E_n E_n^H)

with K the cross-spectral density matrix of the recordings, averaged over time windows, and s
the steering vectors of grid point x. Each returns a float64 numpy array with one value per
grid point.
"""

import numpy as np


def make_problem(
    n_sensors=100,
    grid_limit=100.0,
    grid_spacing=2.0,
    source=(50.0, 0.0),
    window_length=100.0,
    sampling_rate=10.0,
    fmin=0.1,
    fmax=1.0,
    medium_velocity=3.0,
    noise_level=0.0,
    n_windows=None,
    seed=42,
):
    """Sensors in [-25, 25]^2 km, one source, a regular grid of candidate sources.

    With n_windows, the source emits a new random signal in each time window, and
    noise is drawn independently per window; spectra then have shape
    (n_windows, n_sensors, n_freqs).

    Returns a dict of numpy arrays:
        stations         (n_sensors, 2)       sensor coordinates [km]
        gridpoints       (n_gridpoints, 2)    candidate source coordinates [km]
        omega            (n_freqs,)           angular frequencies within [fmin, fmax]
        spectra          (n_sensors, n_freqs) complex spectra of the recordings
                         or (n_windows, n_sensors, n_freqs)
        medium_velocity  float                [km/s]
    """
    rng = np.random.default_rng(seed)
    stations = rng.uniform(-25, 25, size=(n_sensors, 2))

    grid_coords = np.arange(-grid_limit, grid_limit, grid_spacing)
    xx, yy = np.meshgrid(grid_coords, grid_coords, indexing="ij")
    gridpoints = np.stack([xx.ravel(), yy.ravel()], axis=1)

    n_samples = int(window_length * sampling_rate)
    freqs = np.fft.fftfreq(n_samples, 1 / sampling_rate)
    freq_idx = np.where((freqs > fmin) & (freqs < fmax))[0]
    omega = 2 * np.pi * freqs[freq_idx]

    # a flat source spectrum in the band keeps the problem independent of wavelet choice
    traveltimes = np.linalg.norm(stations - np.asarray(source), axis=1) / medium_velocity
    spectra = np.exp(-1j * omega[None, :] * traveltimes[:, None])
    if n_windows is not None:
        # random source phase per window and frequency, i.e. a random source signal
        source_phase = np.exp(2j * np.pi * rng.uniform(size=(n_windows, 1, len(omega))))
        spectra = source_phase * spectra[None]
    if noise_level > 0:
        noise = rng.normal(size=spectra.shape) + 1j * rng.normal(size=spectra.shape)
        spectra = spectra + noise_level * noise / np.sqrt(2)

    return dict(
        stations=stations,
        gridpoints=gridpoints,
        omega=omega,
        spectra=spectra,
        medium_velocity=medium_velocity,
    )


def problem_size(problem):
    n_gridpoints = problem["gridpoints"].shape[0]
    n_sensors, n_freqs = problem["spectra"].shape[-2:]
    return n_gridpoints, n_sensors, n_freqs
