import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from scipy import stats
import scipy.interpolate as interpolate
import scipy.signal
from scipy.linalg import hankel


WHEEL_DIAMETER = 2 * 3.1
WHEEL_TICKS_PER_REV = 1024

def reset_wheel_pos(traj,reset_idx=0):
    """ Function that resets the wheel pos"""
    return traj - traj[reset_idx]

def find_duplicates(array):
    seen = {}
    duplicates = []
    array = np.asarray(array)
    if len(array) == len(set(array)):
        return None
    else:
        for x in array:
            if x not in seen:
                seen[x] = 1
            else:
                duplicates.append(x)
        return duplicates

def find_reaction_point(arr,start_idx=0,end_idx=-1,window_T=3,delta=1):
    """ Finds the reaction point by linear fitting between line segments
        :param arr       : 1-D array to find the reaction point in
        :param start_idx : start index of the interval in the array
        :param end_idx   : start index of the interval in the array
        :param window_T  : half-width of the line segments
        :param delta     : delta step between two line segments
    """
    analysis_range = np.arange(start_idx,end_idx)
    path_segments = []

    for t in analysis_range:
        to_add = arr[t-window_T:t+window_T]
        if len(to_add)==2*window_T:
            path_segments.append([t,to_add])
    
    slope = [0,0]
    for i in range(len(path_segments)-1):
        try:
            xt = path_segments[i][1].reshape(-1,1)
            xt_delta = path_segments[i+delta][1]
        except:
            continue
        try:
            model = LinearRegression().fit(xt,xt_delta)
            slope = [path_segments[i][0],model.coef_[0]] if model.coef_[0]>slope[1] else slope
        except:
            print(arr)
            print(path_segments)
            raise ValueError('kkk')
    return slope

def reset_times(trial_in,cols_to_reset,anchor):
    """ Resets the times in gicen columns depending on a selected reset point(anchor), making it the 0 point
        Basically subtracts the anchor point from all others to make have a time scale that is relative to the anchor
        :param trial_in      : Dict of trial to have their time values reset
        :param cols_to_reset : column names to apply time resetting
        :param anchor        : anchor point in which the other desired columns are reset to
        :type trial_in       : dict
        :type cols_to_reset  : list
        :type anchor         : string
        :return              : Dict with times reset
        :rtype               : dict
    """
    if anchor not in cols_to_reset:
        raise ValueError('Anchor {0} not in'.format(anchor))

    if anchor == 'trialdur':
        time_offset = trial_in['trialdur'][0]
    elif anchor == 'reward':
        if len(trial_in['reward']):
            time_offset = trial_in['reward'][0][0]
        else:
            time_offset = trial_in['closedloopdur'][1] +500
    elif anchor == 'openloopstart':
        time_offset = trial_in['openloopstart']

    for col in cols_to_reset:
        if col in trial_in.keys():
            if col == 'wheel':
                if len(trial_in['wheel']):
                    x = trial_in['wheel']
                    time_synced = np.add(x[:,0], -1*time_offset).reshape(-1,1)
                    # make wheel position start from 0
                    # pos_synced = np.add(x[:,1], -1*x[0,1]).reshape(-1,1)
                    trial_in['wheel'] = np.hstack((time_synced,np.array(x[:,1]).reshape(-1,1)))
            elif col =='lick':
                if len(trial_in['lick']):
                    x = trial_in['lick']
                    time_synced = np.add(x[:,0], -1*time_offset).reshape(-1,1)
                    # change lick counts to trial number for ease of plotting
                    # lick_synced = np.array([trial_in['trial_no']] * len(x)).reshape(-1,1)
                    trial_in['lick'] = np.hstack((time_synced,np.array(x[:,1]).reshape(-1,1)))
            elif col == 'reward':
                if len(trial_in['reward']):
                    x = trial_in['reward']
                    time_synced = np.add(x[:,0], -1*time_offset).reshape(-1,1)
                    # change lick counts to trial number for ease of plotting
                    trial_in['reward'] = np.hstack((time_synced,np.array(x[:,1]).reshape(-1,1)))
            else:
                try:
                    trial_in[col] = np.add(trial_in[col],-1*time_offset)
                except:
                    print(col)
                    print(trial_in[col])
        else:
            print('Column name {0} not in trial data keys. Skipping time reset'.format(col))

    return trial_in

def get_turning_points(x):
    N = 0
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] and x[i + 1] < x[i]:
            N += 1
        elif x[i - 1] > x[i] and x[i + 1] > x[i]:
            N += 1
    return N

def isMonotonic(A):
    """ Returns true if array is monotonic"""
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1)))

def get_wheel_t_range(wheel_time) -> list:
    """ """
    max_t = 0
    min_t = 0
    for wheel_trial in wheel_time:
        if wheel_trial is not None:
            wheel_max = np.max(wheel_trial)
            wheel_min = np.min(wheel_trial)

            max_t = wheel_max if wheel_max > max_t else max_t
            min_t = wheel_min if wheel_min < min_t else min_t

    return [min_t, max_t]

def get_trajectory_avg(wheel_time:np.ndarray,wheel_pos:np.ndarray,n_interp:int=3000,time_range:list=None) -> np.ndarray:
    """ Averages a group of 2D wheel trajectory arrays """
    wheel_stats = {}
    
    if time_range is None:
        wheel_t_range = get_wheel_t_range(wheel_time)
    else:
        wheel_t_range = time_range

    if wheel_time is not None:
        t = np.linspace(wheel_t_range[0], wheel_t_range[1],n_interp).reshape(1,-1)
        # rows = trials
        # columns = wheel_pos points(t)
        wheel_all = np.zeros((len(wheel_time),n_interp))
        wheel_all[:] = np.nan

        for i, wheel_traj in enumerate(wheel_pos):
            if wheel_traj is not None:
                wheel_interp = np.interp(t,wheel_time[i],wheel_traj,left=np.nan,right=np.nan).reshape(1,-1)
                wheel_all[i, :] = wheel_interp
        
        # wheel traj has time on rows, [time, wheel_pos] on columns and trials on depth
        #get the mean 
        avg = np.nanmean(wheel_all, axis=0)
        
        sems_array = stats.sem(wheel_all, axis=0,nan_policy='omit')
        
        wheel_stats['indiv'] = wheel_all
        wheel_stats['avg'] = np.vstack((t, avg))
        wheel_stats['sem'] = np.vstack((t,sems_array.data))
        return t[0,:],wheel_stats

    else:
        return None

def normalized_diff(hit_trials,miss_trials,column):
    """"""
    
    hit = np.array(hit_trials[column])
    miss = np.array(miss_trials[column])
    norm_diff = (np.nanmean(miss) - np.nanmean(hit)) / (np.nanmean(miss) + np.nanmean(hit))
    return float(norm_diff)

def calc_EP(data_in):
    # Error prediction (EP) index. 
    """ By comparing reaction speed, running speed and anticipatory licking 
    in hit versus miss trials, an index of error prediction that reflected 
    whether animals showed reduced response certainty and reward anticipation 
    in incorrect trials. Such a reduction would indicate that an animal had in fact 
    internalized the task rule and was therefore able to predict whether or not a 
    response was correct and therefore likely to result in reward.
    (Havenith et al. 2018 Scientific Reports)
    """
    # data_wlick = data_in[data_in['avg_lick_t_diff']!=-1]
    hit_trials = data_in[data_in['answer']==1]
    miss_trials = data_in[data_in['answer']!=1]

    norm_diffs = []
    for col in ['reaction_t','path_surplus','avg_lick_t_diff']:

        temp = normalized_diff(hit_trials,miss_trials,col)
        print(col, temp)
        if temp is not None:
            norm_diffs.append(temp)
        else:
            print('None received in calculation skipping {0}'.format(col))

    return float(np.nanmean(norm_diffs))


# =====================================================================
# FROM IBL CODEBASE
# =====================================================================


def cm_to_deg(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to degrees turned.  This may be useful for e.g. calculating velocity
    in revolutions per second
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel positions in degrees turned

    # Example: Convert linear cm to degrees
    >>> cm_to_deg(3.142 * WHEEL_DIAMETER)
    360.04667846020925

    # Example: Get positions in deg from cm for 5cm diameter wheel
    >>> import numpy as np
    >>> cm_to_deg(np.array([0.0270526 , 0.04057891, 0.05410521, 0.06763151]), wheel_diameter=5)
    array([0.61999992, 0.93000011, 1.24000007, 1.55000003])
    """
    return positions / (wheel_diameter * np.pi) * 360

def cm_to_rad(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to radians.  This may be useful for e.g. calculating angular velocity.
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel angle in radians

    # Example: Convert linear cm to radians
    >>> cm_to_rad(1)
    0.3225806451612903

    # Example: Get positions in rad from cm for 5cm diameter wheel
    >>> import numpy as np
    >>> cm_to_rad(np.array([0.0270526 , 0.04057891, 0.05410521, 0.06763151]), wheel_diameter=5)
    array([0.01082104, 0.01623156, 0.02164208, 0.0270526 ])
    """
    return positions * (2 / wheel_diameter)

def samples_to_cm(positions, wheel_diameter=WHEEL_DIAMETER, resolution=WHEEL_TICKS_PER_REV):
    """
    Convert wheel position samples to cm linear displacement.  This may be useful for
    inter-converting threshold units
    :param positions: array of wheel positions in sample counts
    :param wheel_diameter: the diameter of the wheel in cm
    :param resolution: resolution of the rotary encoder
    :return: array of wheel angle in radians

    # Example: Get resolution in linear cm
    >>> samples_to_cm(1)
    0.004755340442445488

    # Example: Get positions in linear cm for 4X, 360 ppr encoder
    >>> import numpy as np
    >>> samples_to_cm(np.array([2, 3, 4, 5, 6, 7, 6, 5, 4]), resolution=360*4)
    array([0.0270526 , 0.04057891, 0.05410521, 0.06763151, 0.08115781,
           0.09468411, 0.08115781, 0.06763151, 0.05410521])
    """
    return positions / resolution * np.pi * wheel_diameter

def movements(t, pos, freq=1000, pos_thresh=8, t_thresh=.2, min_gap=.1, pos_thresh_onset=1.5,
              min_dur=.05):
    """
    Detect wheel movements.

    Parameters
    ----------
    t : array_like
        An array of evenly sampled wheel timestamps in absolute seconds
    pos : array_like
        An array of evenly sampled wheel positions
    freq : int
        The sampling rate of the wheel data
    pos_thresh : float
        The minimum required movement during the t_thresh window to be considered part of a
        movement
    t_thresh : float
        The time window over which to check whether the pos_thresh has been crossed
    min_gap : float
        The minimum time between one movement's offset and another movement's onset in order to be
        considered separate.  Movements with a gap smaller than this are 'stictched together'
    pos_thresh_onset : float
        A lower threshold for finding precise onset times.  The first position of each movement
        transition that is this much bigger than the starting position is considered the onset
    min_dur : float
        The minimum duration of a valid movement.  Detected movements shorter than this are ignored

    Returns
    -------
    onsets : np.ndarray
        Timestamps of detected movement onsets
    offsets : np.ndarray
        Timestamps of detected movement offsets
    peak_amps : np.ndarray
        The absolute maximum amplitude of each detected movement, relative to onset position
    peak_vel_times : np.ndarray
        Timestamps of peak velocity for each detected movement
    """
    # Wheel position must be evenly sampled.
    dt = np.diff(t)
    assert np.all(np.abs(dt - dt.mean()) < 1e-10), 'Values not evenly sampled'

    # Convert the time threshold into number of samples given the sampling frequency
    t_thresh_samps = int(np.round(t_thresh * freq))
    max_disp = np.empty(t.size, dtype=float)  # initialize array of total wheel displacement

    # Calculate a Hankel matrix of size t_thresh_samps in batches.  This is effectively a
    # sliding window within which we look for changes in position greater than pos_thresh
    BATCH_SIZE = 10000  # do this in batches in order to keep memory usage reasonable
    c = 0  # index of 'window' position
    while True:
        i2proc = np.arange(BATCH_SIZE) + c
        i2proc = i2proc[i2proc < t.size]
        w2e = hankel(pos[i2proc], np.full(t_thresh_samps, np.nan))
        # Below is the total change in position for each window
        max_disp[i2proc] = np.nanmax(w2e, axis=1) - np.nanmin(w2e, axis=1)
        c += BATCH_SIZE - t_thresh_samps
        if i2proc[-1] == t.size - 1:
            break

    moving = max_disp > pos_thresh  # for each window is the change in position greater than our threshold?
    moving = np.insert(moving, 0, False)  # First sample should always be not moving to ensure we have an onset
    moving[-1] = False  # Likewise, ensure we always end on an offset

    onset_samps = np.where(~moving[:-1] & moving[1:])[0]
    offset_samps = np.where(moving[:-1] & ~moving[1:])[0]
    too_short = np.where((onset_samps[1:] - offset_samps[:-1]) / freq < min_gap)[0]
    for p in too_short:
        moving[offset_samps[p]:onset_samps[p + 1] + 1] = True

    onset_samps = np.where(~moving[:-1] & moving[1:])[0]
    onsets_disp_arr = np.empty((onset_samps.size, t_thresh_samps))
    c = 0
    cwt = 0
    while onset_samps.size != 0:
        i2proc = np.arange(BATCH_SIZE) + c
        icomm = np.intersect1d(i2proc[:-t_thresh_samps - 1], onset_samps, assume_unique=True)
        itpltz = np.intersect1d(i2proc[:-t_thresh_samps - 1], onset_samps,
                                return_indices=True, assume_unique=True)[1]
        i2proc = i2proc[i2proc < t.size]
        if icomm.size > 0:
            w2e = hankel(pos[i2proc], np.full(t_thresh_samps, np.nan))
            w2e = np.abs((w2e.T - w2e[:, 0]).T)
            onsets_disp_arr[cwt + np.arange(icomm.size), :] = w2e[itpltz, :]
            cwt += icomm.size
        c += BATCH_SIZE - t_thresh_samps
        if i2proc[-1] >= onset_samps[-1]:
            break

    has_onset = onsets_disp_arr > pos_thresh_onset
    A = np.argmin(np.fliplr(has_onset).T, axis=0)
    onset_lags = t_thresh_samps - A
    onset_samps = onset_samps + onset_lags - 1
    onsets = t[onset_samps]
    offset_samps = np.where(moving[:-1] & ~moving[1:])[0]
    offsets = t[offset_samps]

    durations = offsets - onsets
    too_short = durations < min_dur
    onset_samps = onset_samps[~too_short]
    onsets = onsets[~too_short]
    offset_samps = offset_samps[~too_short]
    offsets = offsets[~too_short]

    moveGaps = onsets[1:] - offsets[:-1]
    gap_too_small = moveGaps < min_gap
    if onsets.size > 0:
        onsets = onsets[np.insert(~gap_too_small, 0, True)]  # always keep first onset
        onset_samps = onset_samps[np.insert(~gap_too_small, 0, True)]
        offsets = offsets[np.append(~gap_too_small, True)]  # always keep last offset
        offset_samps = offset_samps[np.append(~gap_too_small, True)]

    # Calculate the peak amplitudes -
    # the maximum absolute value of the difference from the onset position
    peaks = (pos[m + np.abs(pos[m:n] - pos[m]).argmax()] - pos[m]
             for m, n in zip(onset_samps, offset_samps))
    peak_amps = np.fromiter(peaks, dtype=float, count=onsets.size)
    N = 10  # Number of points in the Gaussian
    STDEV = 1.8  # Equivalent to a width factor (alpha value) of 2.5
    gauss = scipy.signal.windows.gaussian(N, STDEV)  # A 10-point Gaussian window of a given s.d.
    vel = scipy.signal.convolve(np.diff(np.insert(pos, 0, 0)), gauss, mode='same')
    # For each movement period, find the timestamp where the absolute velocity was greatest
    peaks = (t[m + np.abs(vel[m:n]).argmax()] for m, n in zip(onset_samps, offset_samps))
    peak_vel_times = np.fromiter(peaks, dtype=float, count=onsets.size)
    
    return onsets, offsets, onset_samps, offset_samps, peak_amps, peak_vel_times

def interpolate_position(re_ts, re_pos, freq=1000, kind='linear', fill_gaps=None):
    """
    Return linearly interpolated wheel position.

    Parameters
    ----------
    re_ts : array_like
        Array of timestamps
    re_pos: array_like
        Array of unwrapped wheel positions
    freq : float
        frequency in Hz of the interpolation
    kind : {'linear', 'cubic'}
        Type of interpolation. Defaults to linear interpolation.
    fill_gaps : float
        Minimum gap length to fill. For gaps over this time (seconds),
        forward fill values before interpolation
    Returns
    -------
    yinterp : array
        Interpolated position
    t : array
        Timestamps of interpolated positions
    """
    t = np.arange(re_ts[0], re_ts[-1], 1 / freq)  # Evenly resample at frequency
    if t[-1] > re_ts[-1]:
        t = t[:-1]  # Occasionally due to precision errors the last sample may be outside of range.
    yinterp = interpolate.interp1d(re_ts, re_pos, kind=kind)(t)

    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(re_ts) >= fill_gaps)

        for i in gaps:
            yinterp[(t >= re_ts[i]) & (t < re_ts[i + 1])] = re_pos[i]

    return yinterp, t

def velocity_filtered(pos, fs, corner_frequency=20, order=8):
    """
    Compute wheel velocity from uniformly sampled wheel data.

    pos: array_like
        Vector of uniformly sampled wheel positions.
    fs : float
        Frequency in Hz of the sampling frequency.
    corner_frequency : float
       Corner frequency of low-pass filter.
    order : int
        Order of Butterworth filter.

    Returns
    -------
    vel : np.ndarray
        Array of velocity values.
    acc : np.ndarray
        Array of acceleration values.
    """
    sos = scipy.signal.butter(**{'N': order, 'Wn': corner_frequency / fs * 2, 'btype': 'lowpass'}, output='sos')
    vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos)), 0, 0) * fs
    acc = np.insert(np.diff(vel), 0, 0) * fs
    return vel, acc