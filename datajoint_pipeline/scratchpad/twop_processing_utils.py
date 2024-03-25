import numpy as np


def cut_data_windows(ca_data, ca_time, timings, win_half_time):
    start_indices = [np.argwhere(ca_time >= t)[0][0] for t in timings]
    ca_interval = np.mean(np.diff(ca_time, axis=0))
    win_width = int(np.round(win_half_time / ca_interval))

    n_frames = np.shape(ca_data)[1]
    print(start_indices)
    print(ca_interval)
    print(win_half_time)
    print(win_width)
    # sliced_data = np.stack([ca_data[:, st-win_width : st+win_width+1] for st in start_indices])
    sliced_data = [ca_data[:, st-np.minimum(win_width, st) : np.minimum(st+win_width+1, n_frames)] for st in start_indices]
    
    slice_size = (win_width * 2) +1
    if np.shape(sliced_data[0])[1] < slice_size:
        empty_chunk = np.full([np.shape(sliced_data[0])[0], slice_size - np.shape(sliced_data[0])[1]], np.nan)
        sliced_data[0] = np.concatenate((empty_chunk, sliced_data[0]), axis=1)
    elif np.shape(sliced_data[-1])[1] < slice_size:
        empty_chunk = np.full([np.shape(sliced_data[-1])[0], slice_size - np.shape(sliced_data[-1])[1]], np.nan)
        sliced_data[-1] = np.concatenate((sliced_data[-1], empty_chunk), axis=1)

    sliced_data = np.stack(sliced_data)

    t_array = np.arange(-win_width, win_width+1, 1)
    t = t_array * ca_interval

    return sliced_data, t