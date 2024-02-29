from .utils import *
from scipy.interpolate import PchipInterpolator, interp1d
import scipy.signal
from scipy.linalg import hankel

import matplotlib.pyplot as plt

WHEEL_DIAMETER = 2 * 3.1
WHEEL_TICKS_PER_REV = 1024

class WheelTrace:
    """ Wheel Trace class to isolate the wheel traces"""
    def __init__(self,tick_t:np.ndarray=None, tick_pos:np.ndarray=None,interp_freq:int=5) -> None:
        self.wheel_diameter = WHEEL_DIAMETER
        self.wheel_ticks_per_rev = WHEEL_TICKS_PER_REV
        
        self.set_trace_data(tick_t,tick_pos)
        self.interp_freq = interp_freq
        self.interpolator = None
        
        self.tick_t_interp = None
        # position reaction related
        self.tick_pos_interp = None
        self.pos_reaction_t = None
        self.pos_at_reaction = None
        self.pos_decision_t = None
        self.pos_reaction_bias = None
        self.pos_outcome = None
        # speed reaction related
        self.velo = None
        self.velo_interp = None
        self.speed_reaction_t = None
        self.speed_at_reaction = None
        self.speed_decision_t = None
        self.speed_reaction_bias = None
        self.speed_outcome = None
        
        self.onsets = np.array([])
        self.offsets = np.array([])
        self.onset_samps = np.array([])
        self.offset_samps = np.array([])
        
    def set_trace_data(self, tick_t:np.ndarray, tick_pos:np.ndarray) -> None:
        if tick_t is not None:
            self.abs_tick_t = tick_t
            self.abs_tick_pos = tick_pos
    
    def init_trace(self,time_anchor:float) -> None:
        """"""
        # convert to trial time frame
        self.convert_time_frame(time_frame_anchor=time_anchor) 
        # create an interpolator, needs the frame to be converted first
        self.make_interpolator() 
        # make position 0 at t=0
        self.reset_trajectory() 
        # interpolate the sample points
        self.interpolate_position() 
        # gets the velocity and speed from interpolated position
        self.get_filtered_velocity() 
        
    def make_dict_to_log(self) -> dict:
        """ Makes a dictionary from certain attributes to log them """
        return {'wheel_time' : self.abs_tick_t,
                # 'wheel_pos' : self.ticks_to_cm(self.tick_pos).tolist(),
                'wheel_pos' : self.abs_tick_pos,
                'pos_reaction_time' : self.pos_reaction_t,
                'pos_reaction_bias' : self.pos_reaction_bias,
                'pos_decision_time' : self.pos_decision_t,
                'pos_outcome' : self.pos_outcome,
                'speed_reaction_time' : self.speed_reaction_t,
                'speed_bias' : self.speed_reaction_bias,
                'speed_decision_time' : self.speed_decision_t,
                'speed_outcome' : self.speed_outcome,
                'onsets' : self.onsets.tolist(),
                'offsets': self.offsets.tolist()}
        
    def convert_time_frame(self,time_frame_anchor:float) -> None:
        """
        Converts the time frame, usually from general experiment time frame
        to trial time frame using the anchor
        
        Parameters
        ----------
        time_frame_anchor : float
            Anchor point to be used in converting time frame
        """
        self.time_frame_anchor = time_frame_anchor
        self.tick_t = [t - self.time_frame_anchor for t in self.abs_tick_t]
            
        
    def make_interpolator(self,extrapolate:bool=True) -> None:
        """ 
        Creates an interpolator to interpolate positions
        
        Parameters
        ----------
        extrapolate : bool
            Whether to extrapolate out of bounds values
        """
        try:
            self.interpolator = PchipInterpolator(self.tick_t, self.abs_tick_pos, extrapolate=extrapolate)
        except:
            if extrapolate:
                self.interpolator =interp1d(self.tick_t, self.abs_tick_pos, fill_value='extrapolate')
            else:
                self.interpolator =interp1d(self.tick_t, self.abs_tick_pos)
            
    def reset_trajectory(self, reset_point:float=0) -> None:
        """ 
        Resets the positions to make position 0 at t=0 
        """
        if self.interpolator is not None:
            pos_at0 = self.interpolator(reset_point)
            self.tick_pos = [p - pos_at0 for p in self.abs_tick_pos]
                             
    def make_interval_mask(self,time_window:list=None) -> np.ndarray:
        """ Makes a mask to get the interval of interest in the trace"""
        if time_window is None:
            time_window = [self.tick_t_interp[0],self.tick_t_interp[-1]]
        else:
            # also reset the timr frame of the time window
            time_window = [t - self.time_frame_anchor for t in time_window]
        mask = np.where((time_window[0]<self.tick_t_interp) & (self.tick_t_interp<time_window[1]))
        if not len(mask[0]):
            # this can happen if the animal has not moved the wheel at all,
            # so choose the whole trace
            mask = np.arange(0,len(self.tick_t_interp))
        return mask
    
    def select_trace_interval(self,mask:np.ndarray) -> None:
        """ Makes a dict that has the interval of interests for t, pos, speed and movement onsets """
        tmp  =  {'t' : self.tick_t_interp[mask],
                 'pos' : self.tick_pos_interp[mask],
                 'velo' : self.velo_interp[mask]}
        onsets = []
        t_move = []
        pos_move = []
        velo_move = []
        for i,o in enumerate(self.onsets):
            if o >= tmp['t'][0] and o <= tmp['t'][-1]:
                # onset in range of selected traceinterval
                onset_idx,onset_val = find_nearest(tmp['t'],o) #should be that onset_val == o
                assert o == onset_val
                offset_idx,_ = find_nearest(tmp['t'],self.offsets[i])
                
                t_move.append(tmp['t'][onset_idx:offset_idx+1])
                velo_move.append(tmp['velo'][onset_idx:offset_idx+1])
                pos_move.append(tmp['pos'][onset_idx:offset_idx+1])
                onsets.append(o)
                
        self.trace_interval = {**tmp,
                               'onsets':onsets,
                               'pos_movements':pos_move,
                               'velo_movements':velo_move,
                               't_movements':t_move}     

    def interpolate_position(self, reset_point:float=0, fill_gaps=0):
        """
        Interpolate wheel position.

        Parameters
        ----------
        freq : float
            frequency in Hz of the interpolation
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
        t = np.arange(self.tick_t[0], self.tick_t[-1], 1 / self.interp_freq)  # Evenly resample at frequency
        if t[-1] > self.tick_t[-1]:
            t = t[:-1]  # Occasionally due to precision errors the last sample may be outside of range.
        
        if fill_gaps:
            #  Find large gaps and forward fill @fixme This is inefficient
            gaps, = np.where(np.diff(self.tick_t) >= fill_gaps)
            for i in gaps:
                self.interpolator[(t >= self.tick_t[i]) & (t < self.tick_t[i + 1])] = self.tick_pos[i]

        self.tick_t_interp = t
        pos_at0 = self.interpolator(reset_point)
        self.tick_pos_interp = self.interpolator(self.tick_t_interp) - pos_at0
    
    def cm_to_deg(self,positions):
        """
        Convert wheel position to degrees turned.  This may be useful for e.g. calculating velocity
        in revolutions per second

        # Example: Convert linear cm to degrees
        >>> cm_to_deg(3.142 * WHEEL_DIAMETER)
        360.04667846020925
        """
        return positions / (self.wheel_diameter * np.pi) * 360

    def cm_to_rad(self,positions):
        """
        Convert wheel position to radians.  This may be useful for e.g. calculating angular velocity.

        # Example: Convert linear cm to radians
        >>> cm_to_rad(1)
        0.3225806451612903
        """
        return positions * (2 / self.wheel_diameter)

    def ticks_to_cm(self,positions):
        """
        Convert wheel position samples to cm linear displacement.  This may be useful for
        inter-converting threshold units

        """
        return positions / self.wheel_ticks_per_rev * np.pi * self.wheel_diameter
    
    def get_movements(self, pos_thresh=0.03, t_thresh=0.5, min_gap=.1, pos_thresh_onset=1.5, min_dur=.05) -> None:
        """
        Detect wheel movements. Uses interpolated positions

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
        t = self.tick_t_interp
        pos = self.tick_pos_interp
        freq = self.interp_freq
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
            self.onsets = onsets[np.insert(~gap_too_small, 0, True)]  # always keep first onset
            self.onset_samps = onset_samps[np.insert(~gap_too_small, 0, True)]
            self.offsets = offsets[np.append(~gap_too_small, True)]  # always keep last offset
            self.offset_samps = offset_samps[np.append(~gap_too_small, True)]

        # Calculate the peak amplitudes -
        # the maximum absolute value of the difference from the onset position
        self.peaks = (pos[m + np.abs(pos[m:n] - pos[m]).argmax()] - pos[m]
                for m, n in zip(onset_samps, offset_samps))
        self.peak_amps = np.fromiter(self.peaks, dtype=float, count=onsets.size)
        N = 10  # Number of points in the Gaussian
        STDEV = 1.8  # Equivalent to a width factor (alpha value) of 2.5
        gauss = scipy.signal.windows.gaussian(N, STDEV)  # A 10-point Gaussian window of a given s.d.
        vel = scipy.signal.convolve(np.diff(np.insert(pos, 0, 0)), gauss, mode='same')
        
        # For each movement period, find the timestamp where the absolute velocity was greatest
        self.vel_peaks = (t[m + np.abs(vel[m:n]).argmax()] for m, n in zip(onset_samps, offset_samps))
        self.vel_peak_times = np.fromiter(self.vel_peaks, dtype=float, count=onsets.size)
    
    def convert_movement_to_boolean(self) -> np.ndarray:
        """ This function uses the onset/offset times to create 0/1 boolean arrays of non_movement/movement"""
        bool_moov = np.zeros_like(self.tick_t)
        for i,o in enumerate(self.onsets):
            idx_on,_ = find_nearest(self.tick_t,o)
            idx_off,_ = find_nearest(self.tick_t,self.offsets[i])
            bool_moov[idx_on+1:idx_off] = 1
        return bool_moov
        
    def get_filtered_velocity(self, corner_frequency:float=2, order:int=8):
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
            
        """
        sos = scipy.signal.butter(**{'N': order, 'Wn': corner_frequency / self.interp_freq * 2, 'btype': 'lowpass'}, output='sos')
        self.velo = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, self.tick_pos,padlen=len(self.tick_pos)-1)), 0, 0) * self.interp_freq
        self.velo_interp = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, self.tick_pos_interp)), 0, 0) * self.interp_freq
        # self.speed = np.abs(velocity)
        # self.speed_interp = np.abs(interp_velocity)
        # self.acc = np.insert(np.diff(self.vel), 0, 0) * self.interp_freq
    
    @staticmethod
    def classify_reaction_time(react_t:float) -> int:
        """ Classifies the reaction time to outcome """
        if react_t <= 150:
            ret = -1 # early
        elif react_t > 150 and react_t < 1000:
            ret = 1 # hit
        else:
            ret = 0 # miss
        return ret
    
    def get_speed_reactions(self,speed_threshold:float) -> None:
        """ Selects the first data point where the calculated speed is faster than the speed threshold """
        for i,velo_move in enumerate(self.trace_interval['velo_movements']):
            speed_move = np.abs(velo_move)
            faster_idx = np.where(speed_move > speed_threshold)[0]
            if len(faster_idx):
                self.speed_reaction_t = self.trace_interval['t_movements'][i][faster_idx[0]]
                self.speed_at_reaction = speed_move[faster_idx[0]]
                self.speed_decision_t = self.trace_interval['onsets'][i]
                self.speed_reaction_bias = np.sign(self.trace_interval['velo_movements'][i][faster_idx[0]])
                break
        
        if self.speed_reaction_t is not None:
            self.speed_outcome = self.classify_reaction_time(self.speed_reaction_t)
    
    def get_tick_reactions(self,tick_threshold:int) -> None:
        """ Selects the first data point where the recorded ticks is bigger than the tick_threshold """
        for i,movements in enumerate(self.trace_interval['pos_movements']):
            movements = np.abs(movements) # absolute to look at difference easily
            big_move = np.where(movements > tick_threshold)[0]
            if len(big_move):
                self.pos_reaction_t = self.trace_interval['t_movements'][i][big_move[0]]
                self.pos_at_reaction = self.trace_interval['pos_movements'][i][big_move[0]] # not using movements because of absolute operation above
                self.pos_decision_t = self.trace_interval['onsets'][i]
                self.pos_reaction_bias = np.sign(self.trace_interval['pos_movements'][i][big_move[0]])
                break 
            
        if self.pos_reaction_t is not None:
            self.pos_outcome = self.classify_reaction_time(self.pos_reaction_t)               

    def plot_trajectory(self) -> plt.Figure:
        """ A helper method to visualize the trajectory """
        f = plt.figure(figsize=(20,8))
        ax = f.add_subplot(111)
        
        ax2 = ax.twinx()
        
        # plot interpolated pos
        ax.plot(self.tick_t_interp,
                self.tick_pos_interp,
                color='cornflowerblue',linewidth=2,linestyle='--')
        
        # plot interpolated speed
        ax2.plot(self.tick_t_interp,
                 self.velo_interp,
                 color='goldenrod',linewidth=2,linestyle='--')
        
        # plot selected trace pos interval
        ax.plot(self.trace_interval['t'],
                self.trace_interval['pos'],
                color='b',linewidth=3)
        
        # plot selected trace speed interval
        ax2.plot(self.trace_interval['t'],
                self.trace_interval['velo'],
                color='orange',linewidth=3)
        
        
        # plot movements
        for i,pos_move in enumerate(self.trace_interval['pos_movements']):
            # pos
            ax.scatter(self.trace_interval['onsets'][i],
                       pos_move[0],
                       color='teal',marker='o',s=80)
            ax.plot(self.trace_interval['t_movements'][i],
                    pos_move,
                    color='purple',linewidth=5)
            
            # speed
            ax2.scatter(self.trace_interval['onsets'][i],
                        self.trace_interval['velo_movements'][i][0],
                        color='maroon',marker='o',s=80)
            ax2.plot(self.trace_interval['t_movements'][i],
                     self.trace_interval['velo_movements'][i],
                     color='r',linewidth=5)
        
        # plot recorded data points
        ax.scatter(self.tick_t,
                   self.tick_pos,
                   color='k',marker='+',s=40)
        
        # stim onset
        ax.axvline(0, color='k',linewidth=3)
        
        # plot reactions
        if self.pos_reaction_t is not None:
            ax.axvline(self.pos_reaction_t,
                       color='forestgreen',linewidth=3,linestyle=':')
        
        if self.speed_reaction_t is not None:
            ax2.axvline(self.speed_reaction_t,
                        color='lime',linewidth=3,linestyle=':')
        
        ax.set_title("Wheel Trace")
        ax.set_ylabel('Position',fontsize=14)
        ax2.set_ylabel('Velocity',fontsize=14)
        ax.set_xlabel("Time (ms)",fontsize=14)
        ax.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
        
        ax2.tick_params(axis='y', colors='orange')
        ax2.yaxis.label.set_color('orange')
        ax2.spines['right'].set_color('orange')
        
        return f