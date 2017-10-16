import matplotlib.pyplot as plt
import numpy as np
from hsh_signal.signal import localmax, lowpass, cwt_lowpass, even_smooth, seek_left_localmax
from hsh_signal.heartseries import Series, HeartSeries
from detector import Detector


class ZongDetector(Detector):
    """after Zong et al, 2003: An open-source algorithm to detect onset of arterial blood pressure pulses."""

    DET_LEN = 0.13  #: length of SSF convolution window, in sec
    MWIN_LEN = 5.0  #: length of median window, in sec
    MWIN_PERC = 80  #: amplitude percentile to use, [0-100]
    REFR_LEN = 0.2  #: length of refractory period, in sec
    RUNN_THR = 0.5  #: SSF amplitude threshold, relative to smoothed window percentile

    def __init__(self):
        super(ZongDetector, self).__init__()

    def detect(self, ppgr):
        """:param ppgr: hsh_signal.heartseries.Series object that swings negative."""
        self.ppgr = ppgr
        self.fps = ppgr.fps
        fps = self.fps
        self.det_win = int(ZongDetector.DET_LEN * fps)
        self.m_win = int(ZongDetector.MWIN_LEN * fps)
        self.refr_win = int(ZongDetector.REFR_LEN * fps)
        self.compute_detrended()
        self.compute_clipping()
        self.compute_ssf()
        self.compute_wms()
        self.compute_good()
        self.compute_ppgf()

    def compute_detrended(self):
        fps, ppgr = self.fps, self.ppgr

        # smooth a little bit (to avoid noisy double-peaks on SSF spoiling the amplitude threshold heuristic)
        pleth = lowpass(ppgr.x, fps=fps, cf=6.0, tw=0.5)
        # self.removed_noise = ppgr.x - pleth
        self.pleth = pleth

        base = lowpass(pleth, fps=fps, cf=0.4, tw=0.2)
        detr_unfilt = pleth - base

        # TODO: try center freq hypotheses from FFT peaks.
        detr = cwt_lowpass(detr_unfilt, fps=fps, cf=1.0 / 1.15)

        self.detr_unfilt = detr_unfilt
        self.base, self.detr = base, detr

    def compute_clipping(self):
        x = self.detr  # the signal being clipped
        imin, imax = np.where(localmax(-x))[0], np.where(localmax(x))[0]

        # clipping
        # TODO: ensure amin is negative.
        amin, amax = np.median(x[imin]) * 10, np.median(x[imax]) * 10
        self.amin, self.amax = amin + 1e-15, amax - 1e-15

        xc = np.clip(x, a_min=amin, a_max=amax)
        self.clipped = (xc < self.amin) | (xc > self.amax)

    def compute_ssf(self):
        """compute sum slope function."""
        dyk = -np.diff(self.detr)
        dyk = np.insert(dyk, len(dyk), 0)
        duk = np.clip(dyk, a_min=0, a_max=np.inf)
        self.ssf = np.convolve(np.ones(self.det_win), duk, mode='same')
        self.orig_ssf = np.array(self.ssf)

        # blank out SSF where clipped (so `smooth_wms` amplitude does not overshoot)
        self.near_clipped = np.convolve(np.ones(self.det_win), self.clipped, mode='same')
        self.ssf[np.where(self.near_clipped)[0]] *= 0.0

        # note: convolution results in a kind of moving average, which does slight smoothing to improve noise resistance
        # we could still apply another lowpass filter on `ssf`...
        self.ipeaks = np.where(localmax(self.ssf))[0]

    def compute_wms(self):
        """compute smoothed window medians."""
        win_size = self.m_win
        perc = ZongDetector.MWIN_PERC
        ipeaks = self.ipeaks
        ssf = self.ssf
        wms = []
        for ii, i in enumerate(ipeaks):
            si, ei = max(i - win_size // 2, 0), min(i + win_size // 2 + 1, len(ssf) - 1)
            win_peaks = ipeaks[np.where((ipeaks >= si) & (ipeaks < ei))[0]]
            if len(win_peaks) == 0:
                # absolutely no beats in range? (would be strange.)
                if len(wms) > 0:
                    # use previous threshold
                    wm = wms[-1]
                else:
                    # use global median
                    wm = np.percentile(ssf[ipeaks], perc)
            else:
                wm = np.percentile(ssf[win_peaks], perc)
            wms.append(wm)
        wms = np.array(wms)
        self.wms = wms

        self.smooth_wms = even_smooth(ipeaks, wms, len(ssf), fps=self.fps, cf=0.3, tw=0.1)

    def compute_good(self):
        """compute good beats (apply SSF threshold, refractory period)."""
        ssf, ssff = self.ssf, np.nan_to_num(self.ssf)
        rth = ZongDetector.RUNN_THR
        refr_size = self.refr_win

        isgg = np.where(localmax(ssf) & (np.nan_to_num(ssf) > self.smooth_wms * rth))[0]

        iskipped = []

        isgg_refr = []
        for ii, i in enumerate(isgg):
            sip, eip = max(i - refr_size, 0), min(i, len(ssf) - 1)
            sin, ein = max(i + 1, 0), min(i + refr_size + 1, len(ssf) - 1)
            # is there a taller peak in the previous refractory window? skip this one.
            prev_peak = isgg[np.where((isgg >= sip) & (isgg < eip))[0]]
            # if len(prev_peak) > 0:
            #    #print ii,i,'p',prev_peak,(ssff[prev_peak] > ssff[i]), ssff[prev_peak],ssff[i]
            #    pass
            if len(prev_peak) and np.sum(ssff[prev_peak] > ssff[i]):
                iskipped.append(i)
                continue

            # is there a taller peak in the next refractory window? skip this one.
            next_peak = isgg[np.where((isgg >= sin) & (isgg < ein))[0]]
            # if len(next_peak) > 0:
            #    #print ii,i,'n',next_peak,(ssff[next_peak] > ssff[i]), ssff[next_peak],ssff[i]
            #    pass
            if len(next_peak) and np.sum(ssff[next_peak] > ssff[i]):
                iskipped.append(i)
                continue
            isgg_refr.append(i)

        self.iskipped = np.array(iskipped)
        self.ibeats = np.array(isgg_refr)

        # to do: find key points (foot, edge center, peak)
        # to do: fractional beat time

        # use wavelets to denoise feet?

    def compute_ppgf(self):
        fps, x = self.fps, self.detr
        ifeet = seek_left_localmax(x, self.ibeats, fps)
        self.ifeet = ifeet

        if len(ifeet):
            smooth_feet = even_smooth(ifeet, x[ifeet], len(x), fps=fps, cf=2.0, tw=1.0)
        else:
            smooth_feet = np.zeros(len(x))
        self.smooth_feet = smooth_feet

        # ppgf_x = pleth - base - smooth_feet
        # ppgf_x = clip(ppgf_x)

        ppgf_x = x - smooth_feet

        self.ppgf = Series(ppgf_x, fps=fps, lpad=self.ppgr.lpad)
        # self.ppgrf = Series(ppgf_x + self.removed_noise, fps=fps)

    def plot(self):
        t, ppgf = self.ppgf.t, self.ppgf.x
        ssf = self.ssf
        ipeaks = self.ipeaks
        iskipped = self.iskipped
        ibeats = self.ibeats
        rth = ZongDetector.RUNN_THR

        plt.plot(t, [0] * len(t), c='k')

        # plt.plot(t, self.ppgrf.x, c='m')
        plt.plot(t, ppgf, c='b')
        # plt.plot(t, duk, c='y')
        plt.plot(t, ssf, c='r')
        plt.plot(t[ipeaks], self.wms, c='k')
        plt.plot(t, self.smooth_wms * rth, c='g')

        plt.scatter(t[ipeaks], ssf[ipeaks], c='b')
        if len(iskipped):
            plt.scatter(t[iskipped], ssf[iskipped], c='r')
        if len(ibeats):
            plt.scatter(t[ibeats], ssf[ibeats], c='y')
            plt.scatter(t[ibeats], ppgf[ibeats], c='y')

    def plot_detrend(self):
        t = self.ppgf.t
        plt.plot(t, self.pleth, c='k')
        plt.plot(t, self.base, c='b')
        plt.plot(t, self.detr, c='g')
        plt.plot(t, self.smooth_feet, c='r')
        # ipeaks, ibeats, ifeet
        if len(self.ipeaks):
            plt.scatter(t[self.ipeaks], self.detr[self.ipeaks], c='m')
        if len(self.ifeet):
            plt.scatter(t[self.ifeet], self.detr[self.ifeet], c='r')

    def get_result(self):
        fps, x = self.ppgf.fps, self.ppgf.x
        ibeats = self.ibeats
        return HeartSeries(x, ibeats, fps, self.ppgr.lpad)
