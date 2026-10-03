import jax
import jax.numpy as jnp
from itertools import product
import pywt
import timeit
import torch
import numpy as np
import dask.array as da
import dask

n_sensors = 20
grid_limit = 100
grid_spacing = 2
source = [50, 0]
window_length = 100
sampling_rate = 100
medium_velocity = 3
fmin = 0.1
fmax = 1.0
n_runs = 3


def run_pytorch(
    stations,
    grid_limit,
    grid_spacing,
    medium_velocity,
    omega_lim,
    waveform_spectra_lim,
):
    # generate grid points
    grid_coords = torch.arange(-grid_limit, grid_limit, grid_spacing)
    gridpoints = torch.tensor(list(product(grid_coords, repeat=2)))

    distances_to_all_gridpoints = torch.linalg.norm(
        gridpoints[:, None, :] - stations[None, :, :], axis=2
    )
    traveltimes = distances_to_all_gridpoints / medium_velocity
    # Green's functions between all stations and all grid points
    # within selected frequency band
    # G = exp(-iωt)
    greens_functions = torch.exp(
        -1j * omega_lim[None, None, :] * traveltimes[:, :, None]
    )
    # cross-spectral density matrix of Green's functions
    S = greens_functions[:, :, None, :] * greens_functions.conj()[:, None, :, :]
    # cross-spectral density matrix of recordings
    K = waveform_spectra_lim[:, None, :] * waveform_spectra_lim.conj()[None, :, :]
    # exclude auto-correlations, i.e., do "cross-correlation beamforming"
    diag_idxs = torch.arange(K.shape[0])
    zero_spectra = torch.zeros(omega_lim.shape, dtype=torch.cdouble)
    K[diag_idxs, diag_idxs, :] = zero_spectra
    # Compute cross-correlation beampower
    # using einsum, which automatically identifies ideal path
    # this is about 2x faster than numpy einsum from my experience
    beampowers = torch.einsum("xjkw, kjw -> x", S, K).real


def run_jax(    
    stations,
    grid_limit,
    grid_spacing,
    medium_velocity,
    omega_lim,
    waveform_spectra_lim,):
    # generate grid points
    grid_coords = jnp.arange(-grid_limit, grid_limit, grid_spacing)
    gridpoints = jnp.array(list(product(grid_coords, repeat=2)))
    distances_to_all_gridpoints = jnp.linalg.norm(
        gridpoints[:, None, :] - stations[None, :, :], axis=2
    )
    traveltimes = distances_to_all_gridpoints / medium_velocity
    # Green's functions between all stations and all grid points
    # within selected frequency band
    # G = exp(-iωt)
    greens_functions = jnp.exp(-1j * omega_lim[None, None, :] * traveltimes[:, :, None])
    # cross-spectral density matrix of Green's functions
    S = greens_functions[:, :, None, :] * greens_functions.conj()[:, None, :, :]
    # cross-spectral density matrix of recordings
    K = waveform_spectra_lim[:, None, :] * waveform_spectra_lim.conj()[None, :, :]
    # exclude auto-correlations, i.e., do "cross-correlation beamforming"
    diag_idxs = jnp.arange(K.shape[0])
    zero_spectra = jnp.zeros(omega_lim.shape, dtype=jnp.complex64)
    K = K.at[diag_idxs, diag_idxs, :].set(zero_spectra)
    # Compute cross-correlation beampower
    # using einsum, which automatically identifies ideal path
    # this is about 2x faster than numpy einsum from my experience
    beampowers = jnp.einsum("xjkw, kjw -> x", S, K).real


def run_numpy(
    stations,
    grid_limit,
    grid_spacing,
    medium_velocity,
    omega_lim,
    waveform_spectra_lim,
):
    # define source grid to test
    grid_coords = np.arange(-grid_limit, grid_limit, grid_spacing)
    gridpoints = np.array(list(product(grid_coords, repeat=2)))

    distances_to_all_gridpoints = np.linalg.norm(
        gridpoints[:, None, :] - stations[None, :, :], axis=2
    )
    traveltimes = distances_to_all_gridpoints / medium_velocity

    ### BEAMFORMING
    # Green's functions between all stations and all grid points
    # within selected frequency band
    # G = exp(-iωt)
    greens_functions = np.exp(-1j * omega_lim[None, None, :] * traveltimes[:, :, None])

    # cross-spectral density matrix of Green's functions
    S = greens_functions[:, :, None, :] * greens_functions.conj()[:, None, :, :]

    # cross-spectral density matrix of recordings
    K = waveform_spectra_lim[:, None, :] * waveform_spectra_lim.conj()[None, :, :]

    # exclude auto-correlations, i.e., do "cross-correlation beamforming"
    diag_idxs = np.arange(K.shape[0])
    zero_spectra = np.zeros(omega_lim.shape, dtype=np.complex64)
    K[diag_idxs, diag_idxs, :] = zero_spectra

    # Compute cross-correlation beampower
    # using einsum, which automatically identifies ideal path
    # this is about 2x faster than numpy einsum from my experience
    beampowers = np.einsum("xjkw, kjw -> x", S, K).real


def run_dask(
    stations,
    grid_limit,
    grid_spacing,
    medium_velocity,
    omega_lim,
    waveform_spectra_lim,
):
    # define source grid to test
    grid_coords = da.arange(-grid_limit, grid_limit, grid_spacing)
    gridpoints = da.array(list(product(grid_coords, repeat=2)))

    distances_to_all_gridpoints = da.linalg.norm(
        gridpoints[:, None, :] - stations[None, :, :], axis=2
    )
    traveltimes = distances_to_all_gridpoints / medium_velocity

    ### BEAMFORMING
    # Green's functions between all stations and all grid points
    # within selected frequency band
    # G = exp(-iωt)
    greens_functions = da.exp(-1j * omega_lim[None, None, :] * traveltimes[:, :, None])
    # cross-spectral density matrix of Green's functions
    S = greens_functions[:, :, None, :] * greens_functions.conj()[:, None, :, :]
    # cross-spectral density matrix of recordings
    K = waveform_spectra_lim[:, None, :] * waveform_spectra_lim.conj()[None, :, :]
    # exclude auto-correlations, i.e., do "cross-correlation beamforming"
    diag_idxs = da.arange(K.shape[0])
    zero_spectra = da.zeros(omega_lim.shape, dtype=np.complex64)
    K[diag_idxs, diag_idxs, :] = zero_spectra
    # Compute cross-correlation beampower
    # using einsum, which automatically identifies ideal path
    # this is about 2x faster than numpy einsum from my experience
    beampowers = da.einsum("xjkw, kjw -> x", S, K).real
    beampowers = beampowers.compute()


def prepare_inputs(source, n_sensors, window_length, sampling_rate, medium_velocity):
    np.random.seed(42)
    stations = np.random.uniform(low=-25, high=25, size=(n_sensors, 2))
    source = np.array(source)
    times = np.arange(0, window_length, 1 / sampling_rate)
    freqs = np.fft.fftfreq(len(times), 1 / sampling_rate)
    omega = 2 * np.pi * freqs

    # compute travel times
    distances = np.linalg.norm(stations - source, axis=1)
    traveltimes = distances / medium_velocity

    # define source wavelet
    # wl = ricker(len(times), sampling_rate)
    wavelet = pywt.ContinuousWavelet("cmor0.5-1.0")
    wl = wavelet.wavefun(length=len(times))[0]
    wl = np.array(wl)
    wavelet = np.fft.fft(np.fft.fftshift(wl))

    # compute waveforms for all stations for given source
    # Green's functions are exp(-iωt)
    # waveforms are only computed for plotting purposes.
    # In a usual field data application, waveforms already exist
    # and waveform_spectra need to be computed.
    waveform_spectra = wavelet * np.exp(-1j * omega[None, :] * traveltimes[:, None])

    # limit to frequency band of interest for
    # a) speed-up
    # b) focusing on specific frequencies
    freq_idx = np.where((freqs > fmin) & (freqs < fmax))[0]
    omega_lim = omega[freq_idx]
    waveform_spectra_lim = waveform_spectra[:, freq_idx]

    return stations,omega_lim,waveform_spectra_lim

stations, omega_lim, waveform_spectra_lim = prepare_inputs(source, n_sensors, window_length, sampling_rate, medium_velocity)

print(
    f"n_sensors: {n_sensors}, grid_limit: {grid_limit}, grid_spacing: {grid_spacing}, source: {source}, window_length: {window_length}, sampling_rate: {sampling_rate}, medium_velocity: {medium_velocity}"
)

# convert to torch tensors for PyTorch run to benchmark only the beamforming part
stations_torch = torch.tensor(stations)
omega_lim_torch = torch.tensor(omega_lim)
waveform_spectra_lim_torch = torch.tensor(waveform_spectra_lim)
runtime_torch = timeit.timeit(
    "run_pytorch(stations_torch, grid_limit, grid_spacing, medium_velocity, omega_lim_torch, waveform_spectra_lim_torch)",
    globals=globals(),
    number=n_runs,
)
print(f"PyTorch average runtime: {runtime_torch / n_runs:.2f} seconds")

# convert to jax tensors for JAX run to benchmark only the beamforming part
stations_jax = jnp.array(stations)
omega_lim_jax = jnp.array(omega_lim)
waveform_spectra_lim_jax = jnp.array(waveform_spectra_lim)
runtime_jax = timeit.timeit(
    "run_jax(stations_jax, grid_limit, grid_spacing, medium_velocity, omega_lim_jax, waveform_spectra_lim_jax)",
    globals=globals(),
    number=n_runs,
)
print(f"JAX average runtime: {runtime_jax / n_runs:.2f} seconds")

# dask
stations_dask = da.from_array(stations, chunks="auto")
omega_lim_dask = da.from_array(omega_lim, chunks=-1)
waveform_spectra_lim_dask = da.from_array(waveform_spectra_lim, chunks=("auto", -1))
runtime_dask = timeit.timeit(
    "run_dask(stations_dask, grid_limit, grid_spacing, medium_velocity, omega_lim_dask, waveform_spectra_lim_dask)",
    globals=globals(),
    number=n_runs,
)
print(f"Dask average runtime: {runtime_dask / n_runs:.2f} seconds")

runtime_numpy = timeit.timeit(
    "run_numpy(stations, grid_limit, grid_spacing, medium_velocity, omega_lim, waveform_spectra_lim)",
    globals=globals(),
    number=n_runs,
)
print(f"NumPy average runtime: {runtime_numpy / n_runs:.2f} seconds")
