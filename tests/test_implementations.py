"""Every implementation must reproduce the definition, for every beamformer.

    pytest tests/                       # all available implementations
    pytest tests/ -k numba              # a subset

Implementations whose library or GPU is not available are skipped.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "implementations"))
sys.path.insert(0, str(ROOT))

from problem import make_problem  # noqa: E402

CASES = [
    ("numpy_einsum", {}),
    ("numpy_matmul", {}),
    ("torch_matmul", {}),
    ("torch_matmul", {"device": "cuda"}),
    ("jax_matmul", {"device": "cpu"}),
    ("jax_matmul", {"device": "gpu"}),
    ("cupy_matmul", {}),
    ("numba_loops", {}),
]
METHODS = ["crosscorrelation", "bartlett", "mvdr", "music"]
# 1 / (s^H A s) amplifies rounding errors where s^H A s is small: wider tolerance in single precision
TOLERANCE = {"double": 1e-8, "single": 1e-2}


def gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def definition(problem, method, n_sources=1, diagonal_loading=0.01):
    """The formulas of notebook 4, one grid point and one frequency at a time."""
    spectra, omega = problem["spectra"], problem["omega"]
    n_windows, n_sensors, n_freqs = spectra.shape
    distances = np.linalg.norm(problem["gridpoints"][:, None] - problem["stations"][None], axis=2)
    beampowers = np.zeros(len(distances))
    for w in range(n_freqs):
        d = spectra[:, :, w]
        K = sum(np.outer(d[m], d[m].conj()) for m in range(n_windows)) / n_windows
        if method == "crosscorrelation":
            A = K - np.diag(np.diag(K))
        elif method == "mvdr":
            A = np.linalg.inv(K + diagonal_loading * np.trace(K).real / n_sensors * np.eye(n_sensors))
        elif method == "music":
            noise = np.linalg.eigh(K)[1][:, : n_sensors - n_sources]
            A = noise @ noise.conj().T
        else:
            A = K
        for x in range(len(distances)):
            s = np.exp(-1j * omega[w] * distances[x] / problem["medium_velocity"])
            q = (s.conj() @ A @ s).real
            beampowers[x] += 1 / q / n_freqs if method in ("mvdr", "music") else q
    return beampowers


@pytest.fixture(scope="module")
def problem():
    return make_problem(n_sensors=7, grid_spacing=20, noise_level=0.3, n_windows=20)


@pytest.fixture(scope="module")
def references(problem):
    return {method: definition(problem, method) for method in METHODS}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("name,kwargs", CASES, ids=lambda c: str(c) if isinstance(c, str) else "")
def test_matches_definition(name, kwargs, precision, method, problem, references):
    needs_gpu = kwargs.get("device") in ("cuda", "gpu") or name == "cupy_matmul"
    if needs_gpu and not gpu_available():
        pytest.skip("no GPU")
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        pytest.skip(str(error))
    beampowers = module.beamform(**problem, precision=precision, method=method, **kwargs)
    error = np.abs(beampowers - references[method]).max() / np.abs(references[method]).max()
    assert error < TOLERANCE[precision]


def test_loops_match_definition():
    """The loops of notebook 1: a single window, cross-correlation beamformer."""
    problem = make_problem(n_sensors=5, grid_spacing=40, noise_level=0.3)
    beampowers = importlib.import_module("loops").beamform(**problem)
    reference = definition(problem | {"spectra": problem["spectra"][None]}, "crosscorrelation")
    np.testing.assert_allclose(beampowers, reference, rtol=1e-10, atol=1e-10 * np.abs(reference).max())


def test_peak_at_source():
    import numpy_matmul

    problem = make_problem(n_sensors=30, grid_spacing=5, source=(50.0, 0.0))
    beampowers = numpy_matmul.beamform(**problem)
    np.testing.assert_allclose(problem["gridpoints"][np.argmax(beampowers)], [50.0, 0.0])
