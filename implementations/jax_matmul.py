"""jax, s^H A s as a matrix product, compiled with jax.jit. Runs on device="cpu" or "gpu"."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def matrix(d, method, n_sources=1, diagonal_loading=0.01):
    """A, shape (sensors, sensors), from the spectra d (sensors, windows) of one frequency."""
    K = d @ d.conj().T / d.shape[1]  # cross-spectral density matrix, averaged over windows
    n_sensors = len(K)
    if method == "bartlett":
        return K
    if method == "crosscorrelation":
        return K - jnp.diag(jnp.diag(K))  # leave out the auto-correlations
    if method == "mvdr":
        loading = diagonal_loading * jnp.trace(K).real / n_sensors
        return jnp.linalg.inv(K + loading * jnp.eye(n_sensors))
    if method == "music":
        signal = jnp.linalg.eigh(K)[1][:, -n_sources:]  # eigenvectors of the largest eigenvalues
        return jnp.eye(n_sensors) - signal @ signal.conj().T  # E_n E_n^H = 1 - E_s E_s^H
    raise ValueError(f"unknown method {method!r}")


@partial(jax.jit, static_argnames=("method", "n_sources", "cplx"))
def _one_frequency(d, omega, traveltimes, method, n_sources, diagonal_loading, cplx):
    A = matrix(d, method, n_sources, diagonal_loading).astype(cplx)
    phases = omega * traveltimes
    s = jnp.cos(phases) - 1j * jnp.sin(phases)  # steering vectors, (grid points, sensors)
    q = jnp.sum((s.conj() @ A) * s, axis=1).real  # s^H A s
    return 1 / q if method in ("mvdr", "music") else q


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
    device="cpu",
):
    real, cplx = (jnp.float32, jnp.complex64) if precision == "single" else (jnp.float64, jnp.complex128)
    spectra = spectra if spectra.ndim == 3 else spectra[None]  # (windows, sensors, frequencies)
    traveltimes = np.linalg.norm(gridpoints[:, None, :] - stations[None, :, :], axis=2) / medium_velocity

    with jax.default_device(jax.devices(device)[0]):
        traveltimes = jnp.asarray(traveltimes, dtype=real)  # (grid points, sensors)
        spectra = jnp.asarray(spectra, dtype=jnp.complex128)
        beampowers = jnp.zeros(len(gridpoints))
        for w in range(len(omega)):
            beampowers += _one_frequency(
                spectra[:, :, w].T, real(omega[w]), traveltimes, method, n_sources, diagonal_loading, cplx
            )
        beampowers = np.asarray(beampowers)
    return beampowers / len(omega) if method in ("mvdr", "music") else beampowers
