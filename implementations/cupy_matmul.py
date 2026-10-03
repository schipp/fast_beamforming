"""cupy: the numpy code of numpy_matmul.py on an NVIDIA GPU, with np replaced by cp."""

import cupy as cp
import numpy_matmul


def beamform(stations, gridpoints, omega, spectra, medium_velocity, precision="double", **options):
    arrays = (cp.asarray(a) for a in (stations, gridpoints, omega, spectra))
    return cp.asnumpy(numpy_matmul.beamform(*arrays, medium_velocity, precision, xp=cp, **options))
