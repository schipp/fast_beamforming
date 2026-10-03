"""Download and prepare the recordings for notebook 6: the Greenland landslide and seiche of 2023-09-16.

    python download_greenland.py        # writes data/greenland.mseed (a few MB), takes a few minutes

For every station in greenland_traveltimes.npz: 14 hours of the vertical component at 1 Hz (LHZ),
from the first data centre that has it. The instrument response is removed, and the recordings
are resampled to 0.1 Hz. To use it for your own data, change the stations, the times and the
data centres.
"""

from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

HERE = Path(__file__).parent
stations = np.load(HERE / "greenland_traveltimes.npz")["stations"]  # "network.station"
t1, t2 = UTCDateTime("2023-09-16T11:00:00"), UTCDateTime("2023-09-17T01:00:00")
PROVIDERS = ["IRIS", "GEOFON", "RESIF", "ORFEUS", "INGV", "ETH", "BGR", "NCEDC", "SCEDC", "IPGP"]

stream = Stream()
missing = set(stations)
for provider in PROVIDERS:
    if not missing:
        break
    try:
        bulk = [(*name.split("."), "*", "LHZ", t1, t2) for name in sorted(missing)]
        downloaded = Client(provider).get_waveforms_bulk(bulk, attach_response=True)
    except Exception as error:  # this data centre has none of the stations, or does not answer
        print(f"{provider}: {str(error).splitlines()[0]}")
        continue
    for name in sorted(missing):
        network, station = name.split(".")
        st = downloaded.select(network=network, station=station)
        if not st:
            continue
        st = st.select(location=min(tr.stats.location for tr in st))  # one sensor per station
        try:
            st.detrend("linear")
            st.taper(max_percentage=0.05)
            st.remove_response(output="VEL", pre_filt=(0.002, 0.005, 0.04, 0.05))
            st.merge(fill_value=0)
            st.trim(t1, t2, pad=True, fill_value=0)
            st.resample(0.1)  # includes a low-pass filter without phase shift
        except Exception as error:
            print(f"{name}: {error}")
            continue
        st[0].data = st[0].data.astype(np.float32)
        stream += st
        missing.remove(name)
    print(f"{provider}: {len(stream)} of {len(stations)} stations")

(HERE / "data").mkdir(exist_ok=True)
stream.write(str(HERE / "data" / "greenland.mseed"), format="MSEED")
print(f"wrote data/greenland.mseed; not available: {' '.join(sorted(missing)) or 'none'}")
