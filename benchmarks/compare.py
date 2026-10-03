"""Compare other beamforming tools with the pytorch code of this repository, on the same recordings.

    python benchmarks/compare.py                  # writes results/compare.json
    python benchmarks/compare.py --tools obspy pytorch_planewave --sensors 10 30 --windows 1024

Two problems, each solved with every tool as its documentation suggests:

  plane waves  a plane wave (back-azimuth 60°, slowness 0.3 s/km) on a 101 x 101 slowness grid:
               ObsPy array_processing, TwistPy BeamformingArray, pytorch
  sources      a source at (15, -10) km on a 100 x 100 km grid, 3 km/s:
               acoular, beampower, covseisnet, pytorch

"pytorch" is the cross-correlation beamformer s^H K s of notebook 3, with one K per window.

Recordings: 2 windows of --windows samples at 10 Hz (default 1024, 4096 and 16384 samples:
102 s, 410 s and 27 min), a random signal plus noise, beamformed in 0.1-1 Hz. Each tool runs
in its own process; we report the time per window (after a warm-up run, so that compilation
is not included), the error of its maximum, and how similar its map is to that of pytorch.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLING_RATE = 10.0
WINDOW = 2048  # samples per window; set for each run
N_WINDOWS = 2
FMIN, FMAX = 0.1, 1.0
TRUE_SLOWNESS = 0.3 * np.array([np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60))])  # towards the source
TRUE_SOURCE = np.array([15.0, -10.0])
VELOCITY = 3.0

SLOWNESS_AXIS = np.round(np.arange(-0.5, 0.5 + 1e-9, 0.01), 2)
SLOWNESS_GRID = np.stack(np.meshgrid(SLOWNESS_AXIS, SLOWNESS_AXIS, indexing="ij"), -1).reshape(-1, 2)
SPACE_AXIS = np.arange(-50.0, 50.0, 1.0)
SPACE_GRID = np.stack(np.meshgrid(SPACE_AXIS, SPACE_AXIS, indexing="ij"), -1).reshape(-1, 2)


# ---------------------------------------------------------------- recordings


def make_recordings(n_sensors, problem, seed=0):
    """Waveforms (n_sensors, n_samples) of a random signal from the true source, plus noise."""
    rng = np.random.default_rng(seed)
    stations = rng.uniform(-25, 25, size=(n_sensors, 2))
    if problem == "planewave":
        traveltimes = (stations.mean(axis=0) - stations) @ TRUE_SLOWNESS
    else:
        traveltimes = np.linalg.norm(stations - TRUE_SOURCE, axis=1) / VELOCITY
    n = WINDOW * N_WINDOWS
    freqs = np.fft.rfftfreq(n, 1 / SAMPLING_RATE)
    band = (freqs > 0.05) & (freqs < 2.0)
    source = np.zeros(len(freqs), complex)
    source[band] = np.exp(2j * np.pi * rng.uniform(size=band.sum()))
    signal = np.fft.irfft(source * np.exp(-2j * np.pi * freqs * traveltimes[:, None]), n=n, axis=1)
    waveforms = signal / signal.std() + 0.5 * rng.normal(size=signal.shape)
    return stations, waveforms


def windows_of(waveforms):
    return [waveforms[:, i * WINDOW : (i + 1) * WINDOW] for i in range(N_WINDOWS)]


def obspy_stream(stations, waveforms):
    from obspy import Stream, Trace, UTCDateTime
    from obspy.core.util import AttribDict

    stream = Stream()
    for j, (x, y) in enumerate(stations):
        trace = Trace(waveforms[j].astype(np.float64))
        trace.stats.sampling_rate = SAMPLING_RATE
        trace.stats.starttime = UTCDateTime(2020, 1, 1)
        trace.stats.station = f"S{j:04d}"
        trace.stats.coordinates = AttribDict(x=x, y=y, elevation=0.0)
        stream.append(trace)
    return stream


def bandpass(waveforms):
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [FMIN, FMAX], btype="band", fs=SAMPLING_RATE, output="sos")
    return sosfiltfilt(sos, waveforms, axis=-1)


# ---------------------------------------------------------------- tools: plane waves
# Each returns one beampower map per window, on its own grid, and a function that
# tells whether the maximum of a map is at the true plane wave or source.


def beamform_pytorch(windows, traveltimes):
    """Cross-correlation beampower s^H K s for every window; traveltimes (grid points, sensors)."""
    import torch

    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLING_RATE)
    band = (freqs >= FMIN) & (freqs <= FMAX)
    spectra = np.fft.rfft(np.stack(windows) * np.hanning(WINDOW), axis=-1)[:, :, band]
    spectra = torch.as_tensor(spectra, dtype=torch.complex64)  # (windows, sensors, frequencies)
    omega = torch.as_tensor(2 * np.pi * freqs[band], dtype=torch.float32)
    traveltimes = torch.as_tensor(traveltimes, dtype=torch.float32)
    maps = torch.zeros(len(spectra), len(traveltimes), dtype=torch.float64)
    for w in range(len(omega)):
        phases = -omega[w] * traveltimes
        s = torch.polar(torch.ones_like(phases), phases)  # steering vectors, (grid points, sensors)
        for m in range(len(spectra)):
            K = torch.outer(spectra[m, :, w], spectra[m, :, w].conj())
            K.fill_diagonal_(0)  # leave out the auto-correlations
            maps[m] += ((s.conj() @ K) * s).sum(dim=1).real
    return list(maps.numpy())


def run_pytorch_planewave(stations, waveforms):
    traveltimes = (stations.mean(axis=0) - stations) @ SLOWNESS_GRID.T  # (sensors, slowness vectors)
    return beamform_pytorch(windows_of(waveforms), traveltimes.T), SLOWNESS_GRID


def run_obspy(stations, waveforms):
    from obspy.signal.array_analysis import array_processing

    stream = obspy_stream(stations, waveforms)
    maps = []
    start = stream[0].stats.starttime
    array_processing(
        stream, win_len=(WINDOW - 1) / SAMPLING_RATE, win_frac=1.0,
        sll_x=-0.5, slm_x=0.5, sll_y=-0.5, slm_y=0.5, sl_s=0.01, semb_thres=-1e9, vel_thres=-1e9,
        frqlow=FMIN, frqhigh=FMAX, stime=start, etime=start + (WINDOW * N_WINDOWS - 1) / SAMPLING_RATE,
        prewhiten=0, coordsys="xy", timestamp="julsec", method=0,
        store=lambda relpow, abspow, offset: maps.append(relpow.copy()),
    )  # fmt: skip
    # ObsPy's slowness vectors point in the direction of propagation, ours towards the source
    return [m[::-1, ::-1].ravel() for m in maps][:N_WINDOWS], SLOWNESS_GRID


def run_twistpy(stations, waveforms):
    from twistpy.array_processing import BeamformingArray

    stream = obspy_stream(stations, waveforms)
    array = BeamformingArray(coordinates=np.hstack([stations, np.zeros((len(stations), 1))]))
    with contextlib.redirect_stdout(io.StringIO()):
        array.add_data(stream)
        # TwistPy uses steering vectors at one frequency, and a grid of azimuth and velocity
        array.compute_steering_vectors(
            frequency=0.5 * (FMIN + FMAX), intra_array_velocity=(2.0, 10.0, 0.1),
            inclination=(90, 90, 1), azimuth=(0, 358, 2),
        )  # fmt: skip
    periods = WINDOW / SAMPLING_RATE * 0.5 * (FMIN + FMAX)  # window length in dominant periods
    maps = []
    for i in range(N_WINDOWS):
        start = stream[0].stats.starttime + i * WINDOW / SAMPLING_RATE
        maps.append(array.beamforming("BARTLETT", start, (FMIN, FMAX), periods).ravel())
    azimuths, velocities = np.meshgrid(
        np.deg2rad(np.arange(0, 359, 2)), np.arange(2.0, 10.05, 0.1), indexing="ij"
    )
    # azimuth from the x-axis, counter-clockwise, in the direction of propagation
    grid = -np.stack([np.cos(azimuths), np.sin(azimuths)], -1).reshape(-1, 2) / velocities.reshape(-1, 1)
    return maps, grid


# ---------------------------------------------------------------- tools: sources on a grid


def run_pytorch_mfp(stations, waveforms):
    traveltimes = np.linalg.norm(SPACE_GRID[:, None] - stations[None], axis=2) / VELOCITY
    return beamform_pytorch(windows_of(waveforms), traveltimes), SPACE_GRID


def run_acoular(stations, waveforms):
    import acoular

    acoular.config.global_caching = "none"
    mics = acoular.MicGeom(pos_total=np.vstack([stations.T, np.zeros(len(stations))]))
    grid = acoular.RectGrid(x_min=-50, x_max=49, y_min=-50, y_max=49, z=0, increment=1)
    steer = acoular.SteeringVector(
        grid=grid, mics=mics, steer_type="classic", env=acoular.Environment(c=VELOCITY)
    )
    freqs = np.fft.rfftfreq(WINDOW, 1 / SAMPLING_RATE)
    band = np.where((freqs >= FMIN) & (freqs <= FMAX))[0]
    maps = []
    for window in windows_of(waveforms):
        samples = acoular.TimeSamples(data=window.T.copy(), sample_freq=SAMPLING_RATE)
        spectra = acoular.PowerSpectra(source=samples, block_size=WINDOW, window="Hanning")
        spectra.ind_low, spectra.ind_high = int(band[0]), int(band[-1]) + 1
        beamformer = acoular.BeamformerBase(freq_data=spectra, steer=steer, r_diag=True)
        maps.append(beamformer.result[band[0] : band[-1] + 1].sum(axis=0))  # (x, y) order, like our grid
    return maps, SPACE_GRID


def run_beampower(stations, waveforms):
    import beampower

    traveltimes = np.linalg.norm(SPACE_GRID[:, None] - stations[None], axis=2) / VELOCITY
    delays = np.round(traveltimes * SAMPLING_RATE).astype(np.int32)
    delays = (delays - delays.min(axis=1, keepdims=True))[:, :, None]  # (sources, stations, phases)
    weights_phases = np.ones((len(stations), 1, 1), np.float32)
    weights_sources = np.ones((len(SPACE_GRID), len(stations)), np.float32)
    maps = []
    for window in windows_of(bandpass(waveforms)):
        # time-domain delay-and-sum of the waveforms: one beam per source and sample
        beams = beampower.beamform(
            window[:, None, :].astype(np.float32), delays, weights_phases, weights_sources, reduce="none"
        )
        maps.append(np.sum(beams.astype(np.float64) ** 2, axis=1))  # beam power
    return maps, SPACE_GRID


def run_covseisnet(stations, waveforms):
    import covseisnet

    n = len(stations)
    traveltimes = np.linalg.norm(SPACE_GRID[:, None] - stations[None], axis=2) / VELOCITY
    pairs = np.triu_indices(n, k=1)
    with tempfile.TemporaryDirectory() as directory:
        stream = obspy_stream(stations, waveforms)
        for j in range(n):  # covseisnet reads one travel-time grid per station from disk
            np.save(Path(directory) / f"S{j:04d}.npy", traveltimes[:, j].reshape(100, 100, 1))
        grids = covseisnet.traveltime.TravelTime(stream, directory)
    maps = []
    for window in windows_of(bandpass(waveforms)):
        # cross-correlations of all station pairs (zero lag in the middle), as covseisnet expects
        spectra = np.fft.rfft(window, n=2 * WINDOW, axis=1)
        correlations = np.fft.fftshift(
            np.fft.irfft(spectra[pairs[0]] * spectra[pairs[1]].conj(), axis=1), axes=1
        )
        beam = covseisnet.beam.Beam(1, grids)
        beam.calculate_likelihood(correlations.T, SAMPLING_RATE, 0)
        maps.append(beam.likelihood[0].ravel())
    return maps, SPACE_GRID


TOOLS = {
    # name: (problem, function, description)
    "pytorch_planewave": ("planewave", run_pytorch_planewave, "pytorch, plane waves"),
    "obspy": ("planewave", run_obspy, "ObsPy array_processing"),
    "twistpy": ("planewave", run_twistpy, "TwistPy BeamformingArray"),
    "pytorch_mfp": ("mfp", run_pytorch_mfp, "pytorch, sources"),
    "acoular": ("mfp", run_acoular, "acoular BeamformerBase"),
    "beampower": ("mfp", run_beampower, "beampower.beamform"),
    "covseisnet": ("mfp", run_covseisnet, "covseisnet Beam"),
}


def errors(maps, grid, problem):
    """Largest error of the maximum over all windows."""
    best = grid[[np.argmax(m) for m in maps]]
    if problem == "planewave":
        baz = np.rad2deg(np.arctan2(best[:, 0], best[:, 1]))
        true_baz = np.rad2deg(np.arctan2(*TRUE_SLOWNESS))
        slowness = np.linalg.norm(best, axis=1)
        return dict(
            backazimuth_error_deg=float(np.abs((baz - true_baz + 180) % 360 - 180).max()),
            slowness_error_percent=float(100 * np.abs(slowness / np.linalg.norm(TRUE_SLOWNESS) - 1).max()),
        )
    return dict(location_error_km=float(np.linalg.norm(best - TRUE_SOURCE, axis=1).max()))


def run_tool(name, n_sensors, window):
    global WINDOW
    WINDOW = window
    problem, function, _ = TOOLS[name]
    function(*make_recordings(5, problem))  # warm-up: imports, compilation
    stations, waveforms = make_recordings(n_sensors, problem)
    start = time.perf_counter()
    maps, grid = function(stations, waveforms)
    runtime = (time.perf_counter() - start) / N_WINDOWS
    reference, _ = TOOLS["pytorch_" + problem][1](stations, waveforms)
    return dict(
        tool=name,
        n_sensors=n_sensors,
        window_samples=window,
        seconds_per_window=runtime,
        **errors(maps, grid, problem),
        # agreement with the map of pytorch, where both use the same grid
        correlation_with_pytorch=float(np.corrcoef(maps[0], reference[0])[0, 1])
        if grid is SPACE_GRID or grid is SLOWNESS_GRID
        else None,  # fmt: skip
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tools", nargs="*", default=list(TOOLS))
    parser.add_argument("--sensors", nargs="*", type=int, default=[10, 30, 100, 300, 1000])
    parser.add_argument("--windows", nargs="*", type=int, default=[1024, 4096, 16384], help="samples")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per tool and size")
    parser.add_argument("--output", default=str(ROOT / "results" / "compare.json"))
    parser.add_argument("--_worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        print(json.dumps(run_tool(args._worker[0], int(args._worker[1]), int(args._worker[2]))))
        return

    from run import hardware_info

    output = Path(args.output)
    results = json.loads(output.read_text()) if output.exists() else dict(hardware=hardware_info(), runs=[])
    for name, window in [(name, window) for name in args.tools for window in args.windows]:
        results["runs"] = [
            r for r in results["runs"] if (r["tool"], r.get("window_samples")) != (name, window)
        ]
        for n_sensors in args.sensors:
            cmd = [sys.executable, __file__, "--_worker", name, str(n_sensors), str(window)]
            # acoular runs on one core if numpy's OpenBLAS uses threads (it warns about this)
            env = os.environ | ({"OPENBLAS_NUM_THREADS": "1"} if name == "acoular" else {})
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
                line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                record = (
                    json.loads(line[-1])
                    if line
                    else dict(
                        tool=name,
                        n_sensors=n_sensors,
                        window_samples=window,
                        error=proc.stderr[-500:].strip() or f"stopped (exit code {proc.returncode})",
                    )
                )
            except subprocess.TimeoutExpired:
                record = dict(
                    tool=name,
                    n_sensors=n_sensors,
                    window_samples=window,
                    error=f"more than {args.timeout:.0f} s",
                )
            results["runs"].append(record)
            output.write_text(json.dumps(results, indent=1))
            if "error" in record:
                print(
                    f"{name:20s} {window:6d} {n_sensors:5d} sensors: {record['error'].splitlines()[-1]}",
                    flush=True,
                )
                break  # larger problems will not work either
            details = {
                k: round(v, 3)
                for k, v in record.items()
                if k.endswith(("deg", "percent", "km", "pytorch")) and v
            }
            print(
                f"{name:20s} {window:6d} {n_sensors:5d} sensors: {record['seconds_per_window']:9.4f} s per window, "
                f"{details}",
                flush=True,
            )
            if record["seconds_per_window"] > 300:
                break


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "benchmarks"))
    main()
