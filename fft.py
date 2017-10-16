import numpy as np
import matplotlib.pyplot as plt
from hsh_signal.iter import pairwise
from hsh_signal.signal import localmax, lowpass, bpm2hz, hz2bpm, cwt_lowpass


def gauss(x, t_mu, t_sigma):
    a = 1.0 / (t_sigma * np.sqrt(2 * np.pi))
    y = a * np.exp(-0.5 * (x - t_mu)**2 / t_sigma**2)
    return y


# estimate HR for CWT filtering.
#
# consistency checks:
# * maximum should have more power than others (e.g. 6 dB more)
# * frequency should be consistent with IBI median after beatdetection
#
# * median of several sections should be close to the global HR estimate
#   (could be used as a regularizer constraint)

def fft_hr_estimates(ppg_detrended, f_bias=None, sigm_bias=None, a_bias=None):
    """returns: top list of [(power, hz)] -- power is smoothed a bit."""
    X, fps = np.abs(np.fft.fft(ppg_detrended.x)), float(ppg_detrended.fps)
    f = np.arange(len(X)) * fps / len(X)

    if f_bias is not None:
        #norm = np.sqrt(np.sum(X ** 2))
        bias = gauss(f, f_bias, sigm_bias) # * norm * a_bias
        #X += bias
        X *= bias

    Xl = 10.0 * np.log10(X)
    ff = len(X) / fps
    Xsl = lowpass(Xl, fps=ff, cf=0.1 * ff, tw=0.03 * ff)
    MIN_HZ = bpm2hz(20)  #: min heart rate
    MAX_HZ = bpm2hz(180)  #: max heart rate
    imin, imax = int(ff * MIN_HZ), int(ff * MAX_HZ)
    imax = min(imax, len(Xsl) // 2)
    BEST_RANGE = 10.0  #: return peaks up to BEST_RANGE dB below the best
    idxs = np.where(localmax(Xsl[imin:imax]))[0] + imin
    if len(idxs) == 0: return []
    power_hz = [(Xsl[i], f[i]) for i in idxs]
    power_hz.sort(reverse=True)
    best_power_hz = max(power_hz)
    power_th = best_power_hz[0] - BEST_RANGE
    return [(power, hz) for (power, hz) in power_hz if power >= power_th]


def fft_hr_est_rough(ppg_detrended):
    ppgd = ppg_detrended
    fps = ppgd.fps
    STEP_LEN = 10.0  #: length of step and window, in seconds
    ests, fs = [], []
    for ts,te in pairwise(np.arange(0, ppgd.t[-1], STEP_LEN)):
        si,ei = int(ts*fps), int(te*fps)
        #print si,ei
        ppg_slice = ppgd.slice(slice(si,ei))
        power_hz = fft_hr_estimates(ppg_slice)
        if len(power_hz) == 0: continue
        db_bound = (power_hz[0][0]-power_hz[1][0]) if len(power_hz) >= 2 else 20.0
        ests.append((power_hz[0][0], power_hz[0][1], db_bound))
        fs.append(power_hz[0][1])
        #print ests[-1]  # power,hz,db_bound

    return np.median(fs) if len(fs) > 0 else None


def fft_hr(ppg_detrended):
    ppgd = ppg_detrended.copy()
    ppgd.x = cwt_lowpass(ppgd.x, fps=ppgd.fps, cf=3.0)

    # find the median heart rate of slices, to bias towards
    f_bias, sigm_bias = fft_hr_est_rough(ppgd), 0.1 / 2.0
    a_bias = 0.1

    if f_bias is not None:
        # within the bias range, find the peak power
        power_hz = fft_hr_estimates(ppgd, f_bias, sigm_bias, a_bias)
        if len(power_hz) > 0:
            power, hz = power_hz[0]
            return hz

    return None


if __name__ == '__main__':
    from hsh_signal.app_parser import AppData

    ms = AppData.list_measurements()

    import random
    random.seed(43)
    random.shuffle(ms)

    nerr, ntotal = 0, 0
    errs = []
    for ad in ms[:100]:
        try:
            ppgr, ppgt = ad.ppg_raw(), ad.ppg_trend()
        except IndexError:
            continue
        ppgd = ppgr.copy()
        ppgd.x = ppgr.x - ppgt.x

        hf = fft_hr(ppgd)
        hf = hf if hf is not None else 0.0

        try:
            ppg = ad.ppg_parse_beatdetect('getrr')
        except ZeroDivisionError:
            continue  # ZeroDivisionError: integer division or modulo by zero  in scipy.signaltools.detrend() via ppg_beatdetector_v2.py:54
        psnr = ppg.snr()
        ibis = np.diff(ppg.tbeats)
        # ibi_max, ibi_min = np.percentile(ibis, 70), np.percentile(ibis, 30)
        # ibi_rmean = np.mean(ibis[np.where((ibis < ibi_max) & (ibis > ibi_min))[0]])
        # hg = 1.0 / ibi_rmean

        hg = np.median(ibis)


        errs.append(hz2bpm(hf - hg))
        is_error = (np.abs(hf-hg) > bpm2hz(5.0))

        if is_error:
            nerr += 1
            print ad.mf(), hz2bpm(np.abs(hf-hg)), 'fft=',hz2bpm(hf), 'getrr=',hz2bpm(hg),'snr=',psnr
        ntotal += 1

    print 'nerr=',nerr,'ntotal=',ntotal
    print 'errs=',errs
