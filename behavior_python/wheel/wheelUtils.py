import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

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

def get_wheel_t_range(wheel_arr):
    """ """
    max_t = 0
    min_t = 0
    for wheel_trial in wheel_arr:
        if len(wheel_trial):
            wheel_max = np.max(wheel_trial[:,0])
            wheel_min = np.min(wheel_trial[:,0])

            max_t = wheel_max if wheel_max > max_t else max_t
            min_t = wheel_min if wheel_min < min_t else min_t

    return [min_t, max_t]

def get_trajectory_avg(wheel_arr:np.ndarray) -> np.ndarray:
    """ Averages a group of 2D wheel trajectory arrays"""
    wheel_stats = {}
    
    wheel_t_range = get_wheel_t_range(wheel_arr)

    if len(wheel_arr):
        t = np.linspace(wheel_t_range[0], wheel_t_range[1],2000).reshape(-1,1)
        # rows = wheel data points
        # columns = [time, wheel]
        # depth = trials
        wheel_traj = np.zeros((len(t), 2, len(wheel_arr)))
        wheel_traj[:] = np.nan

        for i, trial_wheel in enumerate(wheel_arr):
            if len(trial_wheel):
                wheel_interp = np.interp(t,trial_wheel[:,0],trial_wheel[:,1],left=np.nan,right=np.nan).reshape(-1,1)
                wheel_put = np.hstack((t,wheel_interp))
                wheel_traj[:, :, i] = wheel_put
        
        # wheel traj has time on rows, [time, wheel_pos] on columns and trials on depth
        #get the mean 
        avg = np.nanmean(wheel_traj[:, 1, :], axis=1).reshape(-1, 1)
        
        sems_array = stats.sem(wheel_traj[:,1,:], axis=1,nan_policy='omit').reshape(-1,1)
        
        wheel_stats['avg'] = np.hstack((t, avg))
        wheel_stats['sem'] = np.hstack((t,sems_array.data))
        return wheel_stats

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