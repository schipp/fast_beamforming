"""pytorch, s^H A s as a matrix product (notebook 3). Runs on a GPU with device="cuda"."""

import torch


def matrix(d, method, n_sources=1, diagonal_loading=0.01):
    """A, shape (sensors, sensors), from the spectra d (sensors, windows) of one frequency."""
    d = d.to(torch.complex128)
    K = d @ d.mH / d.shape[1]  # cross-spectral density matrix, averaged over windows
    n_sensors = len(K)
    identity = torch.eye(n_sensors, dtype=K.dtype, device=K.device)
    if method == "bartlett":
        return K
    if method == "crosscorrelation":
        return K - torch.diag(torch.diag(K))  # leave out the auto-correlations
    if method == "mvdr":
        loading = diagonal_loading * torch.trace(K).real / n_sensors
        return torch.linalg.inv(K + loading * identity)
    if method == "music":
        signal = torch.linalg.eigh(K)[1][:, -n_sources:]  # eigenvectors of the largest eigenvalues
        return identity - signal @ signal.mH  # E_n E_n^H = 1 - E_s E_s^H
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
    device="cpu",
):
    real, cplx = (
        (torch.float32, torch.complex64) if precision == "single" else (torch.float64, torch.complex128)
    )
    stations, gridpoints, omega = (
        torch.as_tensor(a, dtype=real, device=device) for a in (stations, gridpoints, omega)
    )
    spectra = torch.as_tensor(spectra, device=device)
    spectra = spectra if spectra.ndim == 3 else spectra[None]  # (windows, sensors, frequencies)
    traveltimes = torch.cdist(gridpoints, stations) / medium_velocity  # (grid points, sensors)
    adaptive = method in ("mvdr", "music")

    beampowers = torch.zeros(len(gridpoints), dtype=torch.float64, device=device)
    for w in range(len(omega)):
        A = matrix(spectra[:, :, w].T, method, n_sources, diagonal_loading).to(cplx)
        phases = -omega[w] * traveltimes
        s = torch.polar(torch.ones_like(phases), phases)  # steering vectors, (grid points, sensors)
        q = ((s.conj() @ A) * s).sum(dim=1).real  # s^H A s
        beampowers += 1 / q if adaptive else q
    beampowers = beampowers / len(omega) if adaptive else beampowers
    return beampowers.cpu().numpy()
