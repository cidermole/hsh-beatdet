import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import detrend
from hsh_signal.signal import lowpass_fft, highpass
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

beat_on_slope_height = 0.5
print("beat_on_slope_height",beat_on_slope_height)

def movingaverage(signal, window_size):
    window = np.ones(int(window_size))/float(window_size)
    mavg = np.convolve(signal, window, 'same')
    mavg[0] = mavg[1]
    mavg[-2] = mavg[-1]
    return mavg

def make_peaks_alternate(data, minindices, maxindices):
    n = np.min((len(maxindices), len(minindices)))
    for i in range(n):
        needchange = True
        while needchange:
            needchange = False
            if i < len(minindices) - 1 and len(
                    np.where((maxindices > minindices[i]) & (maxindices < minindices[i + 1]))[0]) == 0:  # min,min
                minindices = np.delete(minindices, i)
                needchange = True
            if i < len(maxindices) - 1 and len(
                    np.where((minindices > maxindices[i]) & (minindices < maxindices[i + 1]))[0]) == 0:  # max,max
                maxindices = np.delete(maxindices, i + 1)
                needchange = True
    maxindices = maxindices[:len(minindices)]
    return minindices, maxindices

def delete_small_ibis(tbeats, min_rr):
    todelete = True
    while todelete:
        ibi = np.diff(tbeats)
        delidx = 1 + np.where(ibi < min_rr)[0]
        if len(delidx) > 0:
            tbeats = np.delete(tbeats, delidx)
        else:
            todelete = False
    return tbeats

def climb_to_extrema(data, minindices, maxindices, window=10):
    maxindices = [max(i - window, 0) + np.argmax(data[:,1][max(i - window, 0):min(i + window, len(data) - 1)]) for i in maxindices]
    minindices = [max(i - window, 0) + np.argmin(data[:,1][max(i - window, 0):min(i + window, len(data) - 1)]) for i in minindices]
    return minindices, maxindices

min_downslope_amplitude=0.05
print("min_downslope_amplitude",min_downslope_amplitude)
def getrr_v2(data, fps=125.0, min_rr=250, beat_on_slope_height = beat_on_slope_height, min_downslope_amplitude=min_downslope_amplitude, min_slope_width=0.14, convert_to_ms = False, plt=None):
    if data.shape[0] < fps:
        raise Warning("Warning: Tiny data shape", data.shape[0])
    data = np.copy(data)
    if convert_to_ms:
        data[:, 0] *= 1000.0


    firstsecond = (data[int(fps), 0] - data[0, 0])
    if firstsecond < 900:
        raise Warning("Warning: FPS set to " + str(fps) + ", but timeframe of first FPS data points is " + str(
            firstsecond) + "ms (set convert_to_ms flag to convert between seconds and ms)")

    #data[:, 1] = highpass(highpass(highpass(data[:, 1], fps), fps), fps) # highpass detrend
    data[:, 1] = highpass(highpass(data[:, 1], fps, cf=0.1), fps, cf=0.1)
    data[:, 1] = detrend(data[:, 1]) # linear detrend

    # get initial maxima and minima
    #roughsignal = lowpass_fft(data[:, 1], fps, cf=1.8, tw=0.4)
    #roughsignal = lowpass_fft(data[:, 1], fps, cf=2.5, tw=0.4)
    #roughsignal = lowpass_fft(data[:, 1], fps, cf=2.0, tw=0.9)

    #roughsignal = lowpass_fft(data[:, 1], fps, cf=1.9, tw=0.9)
    roughsignal = lowpass_fft(data[:, 1], fps, cf=2.3, tw=0.9)

    minindices = np.where(heartbeat_localmax(-1 * roughsignal))[0]
    maxindices = np.where(heartbeat_localmax(roughsignal))[0]
    filtered = lowpass_fft(data[:, 1], fps, cf=13, tw=0.6)  # cf=3
    #data[:, 1] = filtered

    # ensure we start with min
    i = 0
    while i < len(maxindices) and maxindices[i] < minindices[0]:
        i += 1
    maxindices = maxindices[i:]
    i = len(minindices) - 1
    while i > 1 and minindices[i] > maxindices[-1]:
        i -= 1
    minindices = minindices[:(i + 1)]

    # get rid of min,min or max,max (they should alternate)
    minindices, maxindices = make_peaks_alternate(data, minindices, maxindices)

    # find actual maxima
    window = int((min_rr/2)/1000.0*float(fps)) # min_rr/2 window for searching for actual peaks
    minindices, maxindices = climb_to_extrema(data, minindices, maxindices, window)
    minindices, maxindices = climb_to_extrema(data, minindices, maxindices, window)

    n = min(len(maxindices), len(minindices))
    amplitude = np.mean(filtered[maxindices[:n]] - filtered[minindices[:n]])

    filtered = lowpass_fft(data[:, 1], fps, cf=3, tw=0.6) # more filtering for nicer visualisation

    # get rid of tiny downslopes
    delidx = []
    for i in range(min(len(maxindices) - 1, len(minindices) - 1)):
        downslope_size = data[maxindices[i], 1] - data[minindices[i+1], 1]
        if 1.0/amplitude*downslope_size < min_downslope_amplitude:
            delidx.append(i)
    maxindices = np.delete(maxindices, delidx)
    minindices = np.delete(minindices, np.array(delidx, dtype=int)+1)

    # get rid of min,min or max,max (they should alternate)
    minindices, maxindices = make_peaks_alternate(data, minindices, maxindices)

    # regress downslopes, find beat
    tbeats, idxs = [], []
    for i in range(min(len(maxindices) - 1, len(minindices) - 1)):
        # make sure there's no intermediate trough (if there is, use that one as min, to be safe)
        intermediate_minimum = np.where(heartbeat_localmax(-data[maxindices[i]:minindices[i + 1],1]))[0]
        if len(intermediate_minimum) > 0:
            k = len(intermediate_minimum) - 1
            while intermediate_minimum[k] == minindices[i+1] and k > 0: # last trough that's different from current min
                k -= 1
            minindices[i+1] = max(maxindices[i] + int(fps*min_slope_width), maxindices[i] + intermediate_minimum[k])
        # climb on if min > max
        if data[minindices[i+1],1] > data[maxindices[i],1]:
            maxindices[i] = minindices[i+1]
            while maxindices[i]+1 < len(data) and data[maxindices[i]+1,1] > data[maxindices[i],1]: # climb up
                maxindices[i] += 1
            minindices[i + 1] += int(fps*min_slope_width)
            while minindices[i+1]+1 < len(data) and data[minindices[i+1]+1,1] < data[minindices[i+1],1]: # climb donw
                minindices[i + 1] += 1

        # find midpoint
        ymid = (data[minindices[i + 1], 1] + (data[maxindices[i], 1] - data[minindices[i + 1], 1]) * beat_on_slope_height)
        try:
            idx = maxindices[i] + np.argmin(np.abs(data[maxindices[i]:minindices[i + 1], 1] - ymid))
        except:
            idx = maxindices[i] + int(fps*min_slope_width) - 1
        tbeat = None
        # linearly interpolate around midpoint
        try:
            f = interp1d(data[idx:idx+2, 1], data[idx:idx+2, 0], kind='linear')
            tbeat = f([ymid])[0]
        except:
            pass
        if tbeat == None:
            try:
                f = interp1d(data[idx-1:idx+1, 1], data[idx-1:idx+1, 0], kind='linear')
                tbeat = f([ymid])[0]
            except:
                try:
                    model = LinearRegression().fit(data[idx-1:idx+1, 1].reshape(-1, 1), data[idx-1:idx+1, 0])
                    tbeat = model.predict([ymid])[0]
                except:
                    tbeat = data[idx, 0]

        downslope_size = data[maxindices[i], 1] - data[minindices[i + 1], 1]
        if 1.0 / amplitude * downslope_size > min_downslope_amplitude:
            tbeats.append(tbeat)
            idxs.append(idx)

            # plotting
            if plt != None:
                #model = TheilSenRegressor().fit(data[maxindices[i]:minindices[i+1], 0].reshape(-1,1), data[maxindices[i]:minindices[i+1], 1])
                #plt.plot(data[maxindices[i]:minindices[i+1]+2, 0], model.predict(data[maxindices[i]:minindices[i+1]+2, 0].reshape(-1,1)), c='b', linewidth=2, alpha=0.6)
                plt.scatter(tbeats[-1], ymid, 120, c='r')
    if plt != None:
        plt.plot(data[:, 0], data[:,1], linewidth=2, c='k')
        plt.plot(data[:,0], roughsignal)
        plt.scatter(data[minindices, 0], data[minindices,1], 30, c='g')
        plt.scatter(data[maxindices, 0], data[maxindices, 1], 30, c='r')
        for i in range(min(len(maxindices)-1,len(minindices)-1)):
            plt.plot(data[maxindices[i]:minindices[i+1],0][:-1], np.diff(data[maxindices[i]:minindices[i+1],1]), c='m')


    # delete too small ibi beats
    tbeats = delete_small_ibis(tbeats, min_rr)
    ibi = np.diff(tbeats)

    return ibi, filtered, idxs

def heartbeat_localmax(d):
    """Calculate the local maxima of a heartbeat signal vector (based on script from https://github.com/compmem/ptsa/blob/master/ptsa/emd.py)."""

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = np.r_[-np.inf, d, -np.inf]
    diffScore = np.diff(np.sign(np.diff(diffvec)))

    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions = np.r_[np.nonzero(d[0:-1] != d[1:])[0], len(d) - 1]
    runLengths = np.diff(np.r_[-1, runEndingPositions])
    runStarts = runEndingPositions - runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths > 1]
    realRunStops = runEndingPositions[runLengths > 1]
    realRunLengths = runLengths[runLengths > 1]

    # save only the runs that are local maxima
    maxRuns = (diffScore[realRunStarts] == -1) & (diffScore[realRunStops] == -1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    maxRunMiddles = np.round(realRunStarts[maxRuns] + realRunLengths[maxRuns] / 2.) - 1

    # get all the maxima
    maxima = (diffScore == -2)
    maxima[maxRunMiddles.astype(np.int32)] = True

    return maxima

