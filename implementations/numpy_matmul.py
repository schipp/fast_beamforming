"""numpy, s^H A s as a matrix product (notebook 2): the K-driven beamformer that all others follow.

For every frequency:

    matrix()    the cross-spectral density matrix K of the recordings, turned into the matrix A
                of the chosen beamformer
    beamform()  s^H A s for the steering vectors s of all grid points

With xp=cupy instead of numpy, the same code runs on a GPU (cupy_matmul.py).
"""

import numpy as np


def matrix(d, method, n_sources=1, diagonal_loading=0.01, xp=np):
    """A, shape (sensors, sensors), from the spectra d (sensors, windows) of one frequency."""
    d = d.astype(xp.complex128)
    K = d @ d.conj().T / d.shape[1]  # cross-spectral density matrix, averaged over windows
    n_sensors = len(K)
    if method == "bartlett":
        return K
    if method == "crosscorrelation":
        return K - xp.diag(xp.diag(K))  # leave out the auto-correlations
    if method == "mvdr":
        loading = diagonal_loading * xp.trace(K).real / n_sensors
        return xp.linalg.inv(K + loading * xp.eye(n_sensors))
    if method == "music":
        signal = xp.linalg.eigh(K)[1][:, -n_sources:]  # eigenvectors of the largest eigenvalues
        return xp.eye(n_sensors) - signal @ signal.conj().T  # E_n E_n^H = 1 - E_s E_s^H
    raise ValueError(f"unknown method {method!r}")


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
    xp=np,
):
    real, cplx = (xp.float32, xp.complex64) if precision == "single" else (xp.float64, xp.complex128)
    spectra = spectra if spectra.ndim == 3 else spectra[None]  # (windows, sensors, frequencies)
    traveltimes = xp.linalg.norm(gridpoints[:, None, :] - stations[None, :, :], axis=2) / medium_velocity
    traveltimes, omega = traveltimes.astype(real), omega.astype(real)  # traveltimes: (grid points, sensors)
    adaptive = method in ("mvdr", "music")

    beampowers = xp.zeros(len(gridpoints))
    for w in range(len(omega)):
        A = matrix(spectra[:, :, w].T, method, n_sources, diagonal_loading, xp).astype(cplx)
        # steering vectors exp(-i w t), (grid points, sensors); cos and sin are faster than a complex exp
        phases = omega[w] * traveltimes
        s = xp.cos(phases) - 1j * xp.sin(phases)
        q = xp.sum((s.conj() @ A) * s, axis=1).real  # s^H A s
        beampowers += 1 / q if adaptive else q
    return beampowers / len(omega) if adaptive else beampowers
