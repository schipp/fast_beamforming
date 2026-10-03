"""numpy, s^H A s with einsum, written like the formula (notebook 2). Runs on one core.

The grid is processed in chunks: the steering vectors of all grid points and frequencies
would not fit into memory.
"""

import numpy as np
from numpy_matmul import matrix


def beamform(
    stations,
    gridpoints,
    omega,
    spectra,
    medium_velocity,
    precision="double",
    method="crosscorrelation",
    n_sources=1,
    diagonal_loading=0.01,
    chunk_bytes=2**28,
):
    real, cplx = (np.float32, np.complex64) if precision == "single" else (np.float64, np.complex128)
    spectra = spectra if spectra.ndim == 3 else spectra[None]  # (windows, sensors, frequencies)
    n_sensors, n_freqs = spectra.shape[1:]
    traveltimes = np.linalg.norm(gridpoints[:, None, :] - stations[None, :, :], axis=2) / medium_velocity
    traveltimes = traveltimes.astype(real)
    omega = omega.astype(real)
    adaptive = method in ("mvdr", "music")

    # A for all frequencies, (sensors, sensors, frequencies)
    A = [matrix(spectra[:, :, w].T, method, n_sources, diagonal_loading) for w in range(n_freqs)]
    A = np.stack(A, axis=2).astype(cplx)

    beampowers = np.zeros(len(gridpoints))
    chunk = max(1, chunk_bytes // (16 * n_sensors * n_freqs))
    for x in range(0, len(gridpoints), chunk):
        # steering vectors, (grid points, sensors, frequencies)
        s = np.exp(-1j * omega[None, None, :] * traveltimes[x : x + chunk, :, None])
        q = np.einsum("xjw, jkw, xkw -> xw", s.conj(), A, s, optimize=True).real  # s^H A s
        beampowers[x : x + chunk] = (1 / q).mean(axis=1) if adaptive else q.sum(axis=1)
    return beampowers
