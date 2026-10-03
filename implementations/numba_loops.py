"""numba, s^H A s as loops, compiled to parallel machine code.

The loops read like the definition (notebook 1). Nothing but one steering vector per
thread is stored. A comes from numpy (numpy_matmul.matrix).
"""

import math

import numba
import numpy as np
from numpy_matmul import matrix


@numba.njit(parallel=True, fastmath=True, cache=True)
def _add_quadratic_forms(A_re, A_im, omega, traveltimes, adaptive, beampowers):
    """beampowers[x] += s^H A s (or its inverse), for one frequency"""
    n_sensors = len(A_re)
    for x in numba.prange(len(beampowers)):
        # steering vector s = exp(-i w t), in real and imaginary parts
        s_re = np.empty(n_sensors, A_re.dtype)
        s_im = np.empty(n_sensors, A_re.dtype)
        for j in range(n_sensors):
            s_re[j] = math.cos(omega * traveltimes[x, j])
            s_im[j] = -math.sin(omega * traveltimes[x, j])
        q = 0.0
        for j in range(n_sensors):
            # (A s)_j, then the real part of s_j^* (A s)_j
            re = A_re.dtype.type(0)
            im = A_re.dtype.type(0)
            for k in range(n_sensors):
                re += A_re[j, k] * s_re[k] - A_im[j, k] * s_im[k]
                im += A_re[j, k] * s_im[k] + A_im[j, k] * s_re[k]
            q += s_re[j] * re + s_im[j] * im
        beampowers[x] += 1 / q if adaptive else q


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
):
    real = np.float32 if precision == "single" else np.float64
    spectra = spectra if spectra.ndim == 3 else spectra[None]  # (windows, sensors, frequencies)
    traveltimes = np.linalg.norm(gridpoints[:, None, :] - stations[None, :, :], axis=2) / medium_velocity
    traveltimes = traveltimes.astype(real)  # (grid points, sensors)
    adaptive = method in ("mvdr", "music")

    beampowers = np.zeros(len(gridpoints))
    for w in range(len(omega)):
        A = matrix(spectra[:, :, w].T, method, n_sources, diagonal_loading)
        A_re, A_im = np.ascontiguousarray(A.real, real), np.ascontiguousarray(A.imag, real)
        _add_quadratic_forms(A_re, A_im, real(omega[w]), traveltimes, adaptive, beampowers)
    return beampowers / len(omega) if adaptive else beampowers
