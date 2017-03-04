import numpy as np, time
from savitzky_golay_filter import savgol_filter, strongsmoothing
from scipy.interpolate import interp1d
from scipy.signal.signaltools import detrend, lfilter, decimate
from scipy.signal.filter_design import butter
import scipy.stats
#import bandpass_filter
from hsh_signal.signal import lowpass_fft, highpass

MIN_PEAK_DIST_IN_STDEVS = 0.5 # 0.43 # 0.45
TREND_MAX_WINDOW = 3 # in seconds

SAVGOL_WINDOWSIZE = 15
TREND_SAVGOL_WINDOWSIZE = 61 # 61
SAVGOL_DEGREE = 3

def crossing_bracket_indices(data, minstartidx=0, maxstartidx=None, crossing=0):
    """ Return indices bracketing the zero crossing on either side (closest to argmin) """
    if not maxstartidx or maxstartidx >= len(data):
        maxstartidx = len(data) - 1
    if maxstartidx <= minstartidx:
        return np.array([np.min((minstartidx,maxstartidx)), np.max((minstartidx,maxstartidx))], dtype=int)
    #zeroidx = minstartidx + np.argmin(np.abs(data[minstartidx:maxstartidx]))
    zeroidx = maxstartidx
    while zeroidx > minstartidx and data[zeroidx] > crossing:
        zeroidx -= 1
    
    idxbracket = [zeroidx, zeroidx]
    while data[idxbracket[0]] > crossing and idxbracket[0] > 0:
        idxbracket[0] -= 1
    while data[idxbracket[1]] < crossing and idxbracket[1] < len(data) - 1:
        idxbracket[1] += 1
    if idxbracket[0] == idxbracket[1]:
        if idxbracket[1] < len(data) - 1:
            idxbracket[1] += 1
        elif idxbracket[0] > 1:
            idxbracket[0] -= 1
    return np.array(idxbracket, dtype=int)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(np.insert(x, 0, N*[x[0]]),len(x)+N, N*[x[-1]])) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def remove_double_beats(data, filtered, maxindices, minindices, min_rr=333):
    n = np.min((len(maxindices), len(minindices)))
    for i in range(1, n):
        needchange = True
        while needchange:
            needchange = False
            if i < len(maxindices) and np.abs(data[maxindices[i], 0] - data[maxindices[i-1], 0]) < min_rr:
                # keep higher peak
                if filtered[maxindices[i]] > filtered[maxindices[i-1]]: 
                    maxindices = np.delete(maxindices, i-1)
                else:
                    maxindices = np.delete(maxindices, i)
                    
                if data[minindices[i], 1] < data[minindices[i-1], 1]: 
                    minindices = np.delete(minindices, i-1)
                else:
                    minindices = np.delete(minindices, i)
                # instead of lowest through; keep the one closest to the highest peak
                """
                if len(minindices) > i:
                    if i < len(maxindices) and np.abs(minindices[i]-maxindices[i]) < np.abs(minindices[i-1]-maxindices[i]) and minindices[i] < maxindices[i]:
                        minindices = np.delete(minindices, i-1)
                    elif i < len(minindices): 
                       minindices = np.delete(minindices, i)
                #"""
    return maxindices, minindices

def interpolate_peaks(data, peakidx, filtersize, stdev=300):
    interpolated = np.zeros((len(data),))
    count =  np.ones((len(data),))
    for i in range(len(peakidx)):
        mini = np.max((0, int(peakidx[i]-filtersize/2)))
        maxi = np.min((len(data), int(peakidx[i]+filtersize/2)))
        count[mini:maxi] += np.ones((maxi-mini,))
        interpolated[mini:maxi] += data[peakidx[i]]*np.ones((maxi-mini,))
        #gauss = scipy.stats.norm.pdf(range(mini-peakidx[i], maxi-peakidx[i]), loc=0, scale=stdev)
        #interpolated[mini:maxi] += gauss
    interpolated /= count.astype(float)
    interpolated /= np.std(interpolated)
    interpolated[interpolated<0] = 0
    return interpolated

def getrr_v1(data, fps = 125.0, min_rr = 300, peakfindrange = 200, slopewidthrange = 100, smooth_slope_ycenters=0.5, discard_short_peaks=False, interpolate=True, interp_n=50, convert_to_ms = False, plt = None, getslopes=False):
    if data.shape[0] < fps:
        return [], [], []
    
    data = np.copy(data)
    if convert_to_ms:
        data[:,0] *= 1000.0
        
    if fps >= 250:
        factor = int(fps/250)
        data = data[np.arange(0, len(data), factor), :]
        
    firstsecond = (data[int(fps), 0]-data[0,0])
    if firstsecond < 900:
        raise Warning("Warning: FPS set to "+str(fps)+", but timeframe of first FPS data points is "+str(firstsecond)+"ms (set convert_to_ms flag to convert between seconds and ms)")
        
    data[:,1] /= np.std(data[:,1])
    data[:,1] = detrend(data[:,1])
    
    # better detrending
    #data[:,1] -= strongsmoothing(data[:,1], 4)
    data[:, 1] = highpass(highpass(data[:, 1], fps), fps)  # highpass detrend
    data[:, 1] = lowpass_fft(data[:, 1], fps, cf=3, tw=0.2) #slight noise filtering

    # outlier removal
    mn, mx = np.min(data[:,1]), np.max(data[:,1])
    m = min(abs(mn),abs(mx))
    containthreshold = 0.001
    N = 100
    step = m/float(N)
    for i in range(N):
        n = len(np.where(data[:,1]<-m)[0]) + len(np.where(data[:,1]>m)[0])
        if n > containthreshold*len(data[:,1]):
            break
        m -= step
    mn, mx = -m, m
    data[data[:,1]<mn,1] = mn
    data[data[:,1]>mx,1] = mx
    
    data[:, 1] /= np.std(data[:, 1])
    data[:, 1] = detrend(data[:, 1])

    # savgol peaks may be off (lower than the actual peak) - do a local max to deal with that
    maxdata = np.zeros((len(data),))
    for i in range(len(maxdata)):
        mini = max(0, i-1)
        maxi = min(len(maxdata), i+2)
        maxdata[i] = np.max(data[mini:maxi,1])

    # get initial maxima and minima
    filtered = savgol_filter(data[:,1], SAVGOL_WINDOWSIZE, SAVGOL_DEGREE)
    mintf = np.array(heartbeat_localmax(-1*filtered))
    maxtf = np.array(heartbeat_localmax(filtered))
    
    # get trend of peaks
    hyp_peakidx = np.where(maxtf)[0]
    peakidx = []
    max_window = min_rr
    for p in hyp_peakidx:
        mini = np.max((0, (p-int(fps*max_window/1e3))))
        maxi = np.min((len(maxdata), (p+int(fps*max_window/1e3))))
        m = mini+np.argmax(maxdata[mini:maxi])
        while m<len(maxdata)-1 and maxdata[m+1]>maxdata[m]:
            m += 1
        while m>0 and maxdata[m-1]>maxdata[m]:
            m -= 1
        if m < len(data):
            peakidx.append(m)
    peakidx = np.unique(peakidx)
    f = interp1d(data[peakidx, 0].flatten(), filtered[peakidx].flatten(), bounds_error=False, kind='linear') # quadratic
    macrotrend = f(data[:, 0].flatten())
    macrotrend[np.where(np.isnan(macrotrend))] = np.mean(macrotrend[np.where(~np.isnan(macrotrend))])     
    macrotrend = savgol_filter(macrotrend, TREND_SAVGOL_WINDOWSIZE, SAVGOL_DEGREE)
    
    # delete if not high or low enough
    """"T = macrotrend+MIN_PEAK_DIST_IN_STDEVS*np.std(filtered) # can leave minima - won't matter
    mintf[np.where(maxdata>T)[0]] = False
    T = macrotrend-MIN_PEAK_DIST_IN_STDEVS*np.std(filtered)
    maxtf[np.where(maxdata<T)[0]] = False"""
    
    minindices = np.where(mintf)[0]
    maxindices = np.where(maxtf)[0]
    
    if plt:
        plt.plot(data[:,0], macrotrend-MIN_PEAK_DIST_IN_STDEVS*np.std(filtered))
        #plt.plot(data[:,0], macrotrend+MIN_PEAK_DIST_IN_STDEVS*np.std(filtered), 'k')
        plt.scatter(data[minindices, 0], filtered[minindices], c='g', s=10)
        plt.scatter(data[maxindices, 0], filtered[maxindices], c='b', s=8)
    
    # ensure we start with min
    i = 0
    while i < len(maxindices) and maxindices[i] < minindices[0]:
        i += 1
    maxindices = maxindices[i:]
    i = len(minindices) - 1
    while i > 1 and minindices[i] > maxindices[-1]:
        i -= 1
    minindices = minindices[:(i+1)]
    
    # min,min or max,max
    n = np.min((len(maxindices), len(minindices)))
    for i in range(n):
        needchange = True
        while needchange:
            needchange = False
            if i<len(minindices)-1 and len(np.where((maxindices>minindices[i])&(maxindices<minindices[i+1]))[0]) == 0: # min,min
                minindices = np.delete(minindices, i)
                needchange = True
            if i<len(maxindices)-1 and len(np.where((minindices>maxindices[i])&(minindices<maxindices[i+1]))[0]) == 0: # min,min
                maxindices = np.delete(maxindices, i+1)
                needchange = True
    maxindices = maxindices[:len(minindices)]
                
    # get rid of too small inter-beat intervals
    maxindices, minindices = remove_double_beats(data, filtered, maxindices, minindices, min_rr=min_rr)
    maxindices, minindices = remove_double_beats(data, filtered, maxindices, minindices, min_rr=min_rr)
        
    if plt:
        #plt.plot(data[:,0], data[:,1], '--k', linewidth=3)
        plt.plot(data[:,0], maxdata)
        plt.plot(data[:,0], macrotrend, 'r', linewidth=0.2)
        plt.plot(data[:,0], filtered, 'k', linewidth=0.5)     
        #plt.scatter(data[:,0], data[:,1], s=20, c='b')
       
    mean_slopemid = (np.mean(maxdata[maxindices]) + np.mean(data[minindices, 1]))/2.0
    max_slopeheight = np.mean(maxdata[maxindices]) - np.mean(data[minindices, 1])
    std_slopeheight = np.std(maxdata[maxindices]) + np.std(data[minindices, 1])
        
    slopebeats = []
    slopes = []
    n = np.min((len(maxindices), len(minindices)))
    # loop through each upslope (min->max) and find an exact beat location in its middle
    for i in range(n-1):        
        
        if discard_short_peaks:
            slopeheight = maxdata[maxindices[i]] - data[minindices[i], 1]
            if slopeheight < max_slopeheight - 2*std_slopeheight - MIN_PEAK_DIST_IN_STDEVS*np.std(filtered):
                if plt:
                    print slopeheight,"<",max_slopeheight - 2*std_slopeheight - MIN_PEAK_DIST_IN_STDEVS*np.std(filtered)
                    plt.scatter(data[maxindices[i],0], maxdata[maxindices[i]], 180, c='k')
                continue        
        
        if interpolate:
            # interpolated
            #ymid = (1-smooth_slope_ycenters)*np.mean([filtered[minindices[i]], filtered[maxindices[i]]])
            ymid = smooth_slope_ycenters*mean_slopemid + (1-smooth_slope_ycenters)*np.mean([maxdata[minindices[i]], 0.5*maxdata[maxindices[i]]])
            idxrange = crossing_bracket_indices(data[:,1], minindices[i], maxindices[i], crossing=ymid)
            
            mni = max(0, idxrange[0] - 2)
            mxi = idxrange[1] + 2
            if mxi > len(data)-1:
                mxi = len(data)-1
            ii = np.linspace(data[mni, 0],data[mxi-1, 0], interp_n)
            f = interp1d(data[mni:mxi, 0], data[mni:mxi, 1], kind='linear') # quadratic
            idata = f(ii)
            iidxrange = crossing_bracket_indices(idata, crossing=ymid)
            try:
                k = (idata[iidxrange][-1]-idata[iidxrange][0])/(ii[iidxrange][-1]-ii[iidxrange][0])
            except:
                k = np.float32.max()
            if plt:
                plt.plot(ii, idata, c='m', linewidth=2)
                plt.scatter(ii[iidxrange], idata[iidxrange], s=40, c='c')
                #k = (idata[iidxrange][-1]-idata[iidxrange][0])/(ii[iidxrange][-1]-ii[iidxrange][0])
                #plt.plot([ii[0]-100, ii[0]+200], [idata[0]-100*k, idata[0]+200*k], '--k', linewidth=2)
                
                
            #ymid = (1-smooth_slope_ycenters)*np.mean([filtered[minindices[i]], filtered[maxindices[i]]])                
            ymid = smooth_slope_ycenters*mean_slopemid + (1-smooth_slope_ycenters)*np.mean([maxdata[minindices[i]], 0.5*maxdata[maxindices[i]]])
            x, y = ii[iidxrange], idata[iidxrange]
            #x, y = ii[iidxrange].reshape(-1,1), idata[iidxrange].reshape(-1,1)
            #model = LinearRegression()
            #model.fit(y.reshape(-1,1), x.reshape(-1,1))
            #xmid = model.predict([[ymid]]) # #xmid2 = [x[0]+y[0]*(x[1]-x[0])/(y[1]-y[0])]
            f = interp1d(y.flatten(), x.flatten(), bounds_error=False, kind='linear') # quadratic
            xmid = f([ymid])
            if np.isnan(xmid[0]):
                xmid = [x[0]+y[0]*(x[1]-x[0])/(y[1]-y[0])]
            if xmid < x[0] or xmid >= x[1]:
                xmid = [np.mean([x[0], x[1]])]
            slopebeats.append(xmid[0])
            slopes.append(k)
        else:
            mididx = minindices[i]+(maxindices[i]-minindices[i])/2
            if mididx-int(mididx) > 0.5 or int(mididx) >= len(data) - 1:
                idxrange = [int(mididx)-2, int(mididx)]
            else:
                idxrange = [int(mididx)-1, int(mididx)]        
            x, y = data[idxrange, 0], data[idxrange, 1]
            ymid = (1-smooth_slope_ycenters)*np.mean([filtered[minindices[i]], filtered[maxindices[i]]])
            f = interp1d(y.flatten(), x.flatten(), bounds_error=False, kind='linear') # quadratic
            xmid = f([ymid])
            slopebeats.append(xmid[0])
        
        if plt:
            try:
                plt.scatter(data[idxrange, 0], data[idxrange, 1], s=40, c='y')
                plt.scatter(data[minindices[i], 0], maxdata[minindices[i]], c='g', s=100)
                plt.scatter(data[maxindices[i], 0], maxdata[maxindices[i]], c='b', s=80)
                plt.plot(x, y)
                #yp = data[minindices[i]:maxindices[i], 1]
                #xp = model.predict(yp.reshape(-1,1))
                #plt.plot(xp, yp, '--k', linewidth=2)
                plt.scatter(xmid, ymid, s=80, c='r')
            except Exception,e:
                pass
                #print "PLOT ERROR:",e
    
    slopebeats = np.array(slopebeats)
    # delete unreasonable values
    delidx = np.hstack((np.where(slopebeats<data[0,0])[0], np.where(slopebeats>data[-1,0])[0])) 
    slopebeats = np.delete(slopebeats, delidx)
    slopes = np.delete(slopes, delidx)
    # delete too small ibi beats
    delidx = [True]
    while len(delidx)>0:
        ibi = (slopebeats - np.roll(slopebeats, 1))[1:]
        delidx = 1+np.where(ibi<min_rr)[0]
        slopebeats = np.delete(slopebeats, delidx)
        slopes = np.delete(slopes, delidx)
    
    idx = [np.argmin(np.square(data[:,0]-slopebeats[i])) for i in range(len(slopebeats))]
    ibi = (slopebeats - np.roll(slopebeats, 1))[2:] #2:
    while len(ibi)>0 and np.isnan(ibi[0]):
        ibi = ibi[1:]
       
    """plt.show() 
    plt.plot(data[:,0], data[:, 1])
    plt.scatter(data[idx,0], data[idx, 1], c='r')
    print ibi
    plt.show()"""

    if getslopes:        
        return ibi, filtered, idx, slopes
    else:
        return ibi, filtered, idx

def smoothed_peakdetection(data, fps, min_rr, windowsize=15, degree=3):
    filtered = savgol_filter(data[:,1], windowsize, degree)
    idx = np.array(heartbeat_localmax(filtered))
    T = np.mean(filtered)+MIN_PEAK_DIST_IN_STDEVS*np.std(filtered)
    idx[np.where(filtered<T)[0]] = False
    beats = data[idx, 0]
    
    ibi = (beats - np.roll(beats, 1))
    ibi[0] = np.mean(ibi[1:])
    removeidx = np.where(ibi<min_rr)[0]-1 # remove the first, not the second, for peaks that are too close
    idx[np.where(idx)[0][removeidx]] = False
    beats = data[idx, 0]
    
    indices = np.where(idx)[0]
    return beats, indices, filtered

def nearestpeak(i, data, climbrange):
    leftshift = 0
    rightshift = 0
    try:
        while data[i-leftshift-1] > data[i-leftshift] and leftshift < climbrange:
            leftshift += 1
    except:
        pass
    try:
        while data[i+rightshift+1] > data[i+rightshift] and rightshift < climbrange:
            rightshift += 1
    except:
        pass
    if data[i-leftshift] > data[i+rightshift]:
        return i-leftshift
    else:
        return i+rightshift

def getbestshift(beatibi, ecgibi):
    errs = []
    error_series = []
    for i in range(len(ecgibi)-len(beatibi)-1):
        errs.append(np.mean(np.abs(beatibi.flatten() - ecgibi[i:(i+len(beatibi))].flatten())))
        error_series.append(beatibi.flatten() - ecgibi[i:(i+len(beatibi))].flatten())
    try:
        bestshift = np.argmin(errs)
    except:
        return 0, np.infty
    return bestshift, errs, error_series[bestshift].astype(float)

def heartbeat_localmax(d):
    """Calculate the local maxima of a heartbeat signal vector (based on script from https://github.com/compmem/ptsa/blob/master/ptsa/emd.py)."""

    # this gets a value of -2 if it is an unambiguous local max
    # value -1 denotes that the run its a part of may contain a local max
    diffvec = np.r_[-np.inf,d,-np.inf]
    diffScore=np.diff(np.sign(np.diff(diffvec)))
                     
    # Run length code with help from:
    #  http://home.online.no/~pjacklam/matlab/doc/mtt/index.html
    # (this is all painfully complicated, but I did it in order to avoid loops...)

    # here calculate the position and length of each run
    runEndingPositions=np.r_[np.nonzero(d[0:-1]!=d[1:])[0],len(d)-1]
    runLengths = np.diff(np.r_[-1, runEndingPositions])
    runStarts=runEndingPositions-runLengths + 1

    # Now concentrate on only the runs with length>1
    realRunStarts = runStarts[runLengths>1]
    realRunStops = runEndingPositions[runLengths>1]
    realRunLengths = runLengths[runLengths>1]

    # save only the runs that are local maxima
    maxRuns=(diffScore[realRunStarts]==-1) & (diffScore[realRunStops]==-1)

    # If a run is a local max, then count the middle position (rounded) as the 'max'
    maxRunMiddles=np.round(realRunStarts[maxRuns]+realRunLengths[maxRuns]/2.)-1

    # get all the maxima
    maxima=(diffScore==-2)
    maxima[maxRunMiddles.astype(np.int32)] = True

    return maxima

def butter_pass(cutoff, fs, order=5, btype="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def butter_bandpass_filter(data, cutoffs=[0.5, 3], fs=125.0, order=5): 
    # data, cutoff frequency, sampling frequency, butterworth filter order 
    b, a = butter_pass(cutoffs[1], fs, order=order, btype="low")
    data_lp = lfilter(b, a, data)
    
    b, a = butter_pass(cutoffs[0], fs, order=order, btype="high")
    data_hp = lfilter(b, a, data_lp)
    
    return data_hp
