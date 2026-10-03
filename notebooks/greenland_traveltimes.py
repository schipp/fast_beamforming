"""How the traveltime table of notebook 6 was computed: fast marching through a phase-velocity map.

    python greenland_traveltimes.py lith01_10mHz.txt lith01_15mHz.txt    # writes greenland_traveltimes.npz

For every grid point, the fast marching method (pykonal; pip install pykonal) computes the
traveltime of a Rayleigh wave from the grid point to everywhere on the globe, through a map of
phase velocities; we then read off the traveltimes at the stations. The velocity maps here are
those of LITHO1.0 (Pasyanos et al., 2014) at 10 and 15 mHz, as text files with the columns
longitude (0 to 360), latitude, phase velocity [km/s], on a 1 degree grid.

Every case needs its own table: another velocity model (2D or 3D), ray tracing or a wave
simulation instead of fast marching, other stations and grid points. Whatever the method, the
result is the same kind of table, (grid points, stations), in seconds.

As it is, this script reproduces the table of this repository. One grid point takes a few seconds
(0.2 s with SPACING = 0.5 degrees, which changes the traveltimes by about a second); the 2623 grid
points are spread over all cores.
"""

import sys
from multiprocessing import Pool

import numpy as np
import pykonal
from scipy.interpolate import RegularGridInterpolator

PERIOD = 92.0  # s, of the signal
SPACING = 0.1  # degrees, of the nodes that fast marching runs on
EARTH_RADIUS = 6371.0  # km

# stations (here: those of the existing table) and grid points
table = dict(np.load("greenland_traveltimes.npz"))
grid_lat, grid_lon = np.meshgrid(np.arange(68.5, 77.01, 0.2), np.arange(-38, -13.99, 0.4), indexing="ij")
table["grid_lon"], table["grid_lat"] = (
    grid_lon.ravel().astype(np.float32),
    grid_lat.ravel().astype(np.float32),
)


def load_velocity_map(path):
    """Phase velocities as (latitudes, longitudes), with their latitudes and longitudes (-180 to 180)."""
    lon, lat, velocity = np.loadtxt(path, usecols=(0, 1, 2)).T
    lon = np.where(lon > 180, lon - 360, lon)
    lons, lats = np.unique(lon), np.unique(lat)
    grid = np.empty((len(lats), len(lons)))
    grid[np.searchsorted(lats, lat), np.searchsorted(lons, lon)] = velocity
    return grid, lats, lons


def traveltimes_to_stations(source):
    """Traveltimes from one grid point (lon, lat) to all stations."""
    lon, lat = source
    solver = pykonal.EikonalSolver(coord_sys="spherical")  # coordinates: radius, polar angle, azimuth
    solver.velocity.min_coords = EARTH_RADIUS, 0, 0
    solver.velocity.node_intervals = 1, np.deg2rad(SPACING), np.deg2rad(SPACING)
    solver.velocity.npts = 1, len(node_lats), len(node_lons)  # a single shell: the surface
    solver.velocity.values = node_velocities[None]
    # the source: the node closest to the grid point
    source_idx = 0, np.argmin(np.abs(node_lats - lat)), np.argmin(np.abs(node_lons - lon))
    solver.traveltime.values[source_idx] = 0
    solver.unknown[source_idx] = False
    solver.trial.push(*source_idx)
    solver.solve()
    field = RegularGridInterpolator((node_lats, node_lons), solver.traveltime.values[0])
    return field(np.stack([table["station_lat"], table["station_lon"]], axis=1))


# the velocity map at the period of the signal, between the maps at 10 mHz (100 s) and 15 mHz (66 s)
map_10mHz, lats, lons = load_velocity_map(sys.argv[1])
map_15mHz, _, _ = load_velocity_map(sys.argv[2])
weight = (100 - PERIOD) / (100 - 66)
velocities = RegularGridInterpolator(
    (lats, lons), (1 - weight) * map_10mHz + weight * map_15mHz, bounds_error=False, fill_value=None
)
# ... interpolated to the nodes of fast marching
node_lats, node_lons = np.arange(-90, 90 + SPACING / 2, SPACING), np.arange(-180, 180, SPACING)
node_velocities = velocities(tuple(np.meshgrid(node_lats, node_lons, indexing="ij")))

if __name__ == "__main__":
    with Pool() as pool:
        sources = zip(table["grid_lon"], table["grid_lat"])
        table["traveltimes"] = np.array(pool.map(traveltimes_to_stations, sources), dtype=np.float32)
    np.savez_compressed("greenland_traveltimes.npz", **table)
