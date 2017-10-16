import numpy as np
from ML.ppg_beatdetector_v1 import getrr_v1
from ML.ppg_beatdetector_v2 import getrr_v2
from hsh_signal.heartseries import HeartSeries
from hsh_signal.signal import highpass


def beatdet_getrr_v2(data, get_tbeats=False):
    if get_tbeats:
        ibi, filtered, idxs, tbeats = getrr_v2(data, fps=30, convert_to_ms=True, get_tbeats=True)
        return ibi, filtered, idxs, tbeats
    else:
        ibi, filtered, idxs = getrr_v2(data, fps=30, convert_to_ms=True, get_tbeats=False)
        return ibi, filtered, idxs


def beatdet_getrr_v2_fracidx(data):
    fps=30.0
    ibi, filtered, idxs_whole, tbeats = getrr_v2(data, fps=fps, convert_to_ms=True, get_tbeats=True)
    t0 = data[0,0]
    trel = np.array(tbeats)/1e3 - t0
    idxs = trel * fps
    idxs = idxs[np.where(idxs < data.shape[0])[0]]
    return ibi, filtered, idxs


def beatdet_getrr_v1(data):
    series = data[:,1]
    reversed_data = np.vstack((data[:,0], list(reversed(series)))).T
    ibis, filtered, idx = getrr_v1(reversed_data, fps = 30, convert_to_ms=True)
    ibis = np.array(list(reversed(ibis)))
    idx = list((len(series)-1) - np.array(list(reversed(idx))))
    filtered = np.array(list(reversed(filtered)))
    return ibis, filtered, idx


def beatdet(data, beat_detector, fps=30):
    times, series = data[:,0], data[:,1]

    if np.std(series) < 1e-6:
        # if np.std(series)==0, getrr() will raise "ValueError: array must not contain infs or NaNs"
        # instead, return no beats == []
        return HeartSeries(series, [], fps=fps, lpad=-times[0]*fps)

    ibis, filtered, idx = beat_detector(data)

    hp_series = highpass(highpass(series, fps), fps)

    return HeartSeries(hp_series, idx, fps=fps, lpad=-times[0]*fps)
