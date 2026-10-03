"""Python loops over grid points and sensor pairs: the definition, B(x) = sum_w s^H K s.

Only usable for tiny problems. The reference all other implementations are tested against.
"""

import cmath
import math


def beamform(stations, gridpoints, omega, spectra, medium_velocity, precision="double"):
    n_sensors = len(stations)
    beampowers = []
    for gx, gy in gridpoints:
        # steering vector: the phase delays expected for a source at this grid point
        steering = []
        for sx, sy in stations:
            traveltime = math.hypot(gx - sx, gy - sy) / medium_velocity
            steering.append([cmath.exp(-1j * w * traveltime) for w in omega])

        beampower = 0.0
        for w_idx in range(len(omega)):
            for j in range(n_sensors):
                for k in range(n_sensors):
                    if j == k:
                        continue  # leave out the auto-correlations
                    K_jk = spectra[j][w_idx] * spectra[k][w_idx].conjugate()
                    beampower += (steering[j][w_idx].conjugate() * K_jk * steering[k][w_idx]).real
        beampowers.append(beampower)
    return beampowers
