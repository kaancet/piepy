import numpy as np
from numpy.typing import ArrayLike
import scipy.signal
from scipy.linalg import hankel
from scipy.interpolate import PchipInterpolator

WHEEL_DIAMETER = 2 * 3.1
WHEEL_TICKS_PER_REV = 1024


class WheelTrace:
    _interpolator = None

    def __init__(self):
        pass

    @staticmethod
    def find_nearest(arr: ArrayLike, value: float) -> int:
        """ Returns the index of the nearest

        Args:
            arr (ArrayLike): Array of values to search in
            value (float): Find the nearest value to this one that exists in the input arr

        Returns:
            int: index of the nearest value in the arr
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return np.nanargmin(np.abs(arr - value))
    
    @staticmethod
    def fix_trace_timing(t:np.ndarray,pos:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """ Looks at differences between time points and make sure the difference is always positive(strictly monotonically increasing) 

        Args:
            t (np.ndarray): Time values of the wheel trace
            pos (np.ndarray): Position values of the wheel trace

        Returns:
            tuple[np.ndarray,np.ndarray]: Fixed time and pos values
        """
        while not np.all(np.diff(t) > 0):
            if np.diff(t)[-1] < 0:
                # sometimes the last element is problematic
                t = t[:-1]
                pos = pos[:-1]
            else:
                # find and delete that fucker
                _idx = np.where(np.diff(t) <= 0)[0]
                t = np.delete(t, _idx)
                pos = np.delete(pos, _idx)
        return t,pos

    @classmethod
    def init_interpolator(cls, t: ArrayLike, pos: ArrayLike) -> None:
        """ Initialize the interpolator

        Args:
            t (np.ndarray): Time values of the wheel trace
            pos (np.ndarray): Position values of the wheel trace

        Raises:
            ValueError: If lengths of t and pos are not equal
        """
        if len(t) != len(pos):
            raise ValueError("Unequal wheel time and position!!")

        if len(t) == 1:
            # only single value, means no wheel movement
            # add another point with same pos but incremented t
            t = np.append(t, t[0] + 10)
            pos = np.append(pos, pos[0])

        cls._interpolator = PchipInterpolator(t, pos, extrapolate=True)

    @classmethod
    def reset_time_frame(cls, t: np.ndarray, reset_time_point: float) -> np.ndarray:
        """ Resets the time ticks to be zero at reset_time_point

        Args:
            t (np.ndarray): Time values of the wheel trace
            reset_time_point (float): Time value to reset the trace time values

        Returns:
            np.ndarray: reset time values
        """
        return t - reset_time_point

    @classmethod
    def reset_position(cls, pos: np.ndarray, reset_point: int) -> np.ndarray:
        """ Resets the positions to make position 0 at t=0
        This method is used on the ticks so the values can (and should) be integers

        Args:
            pos (np.ndarray): Position values of the wheel trace
            reset_point (int): Position point to reset the tick values of the trajectory

        Returns:
            np.ndarray: reset position values
        """
        if cls._interpolator is not None:
            pos_at0 = round(cls._interpolator(reset_point).tolist())
            _temp = pos - pos_at0
            return _temp.astype(int)

    @classmethod
    def reset_and_interpolate(
        cls, t: np.ndarray, pos: np.ndarray, reset_time: float, interp_freq: float = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Fundemental loop of resetting time, position and interpolating

        Args:
            t (np.ndarray): Time values of the wheel trace
            pos (np.ndarray): Position values of the wheel trace
            reset_time (float): Time value to reset the trace time values
            interp_freq (float, optional): Interpolation frequency. Defaults to 5.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Reset time and position, interpolated time and position
        """
        if not reset_time > t[0]:
            # means that the first wheel movement was recorded after the given reset time point
            # fill_in the time and pos until reaching the reset_time
            add_t = np.arange(reset_time,t[0],50)
            add_pos = np.array([pos[0]]*len(add_t))
            
            t = np.append(add_t,t)
            pos = np.append(add_pos,pos)

        reset_t = cls.reset_time_frame(t, reset_time)
            
        # init interpolator with ticks first
        cls.init_interpolator(reset_t, pos)
        # reset positions
        reset_tick = cls.reset_position(pos, 0)
        # reinit interpolator
        cls.init_interpolator(reset_t, reset_tick)
        # interpolate the whole trace
        t_interp, tick_interp = cls.interpolate_trace(reset_t, reset_tick, interp_freq)

        return reset_t, reset_tick, t_interp, tick_interp

    @classmethod
    def interpolate_trace(
        cls, t: np.ndarray, pos: np.ndarray, interp_freq: float = 5
    ) -> np.ndarray:
        """ Interpolate wheel positions

        Args:
            t (np.ndarray): Time values of the wheel trace
            pos (np.ndarray): Position values of the wheel trace
            interp_freq (float, optional): Interpolation frequency. Defaults to 5.

        Returns:
            np.ndarray: Interpolated tick values from interpolated time values
        """
        if cls._interpolator is not None:
            if len(t) == 1:
                # only single value, means no wheel movement
                # add another point with same pos but incremented t
                t = np.append(t, t[0] + 10)

            interp_t = np.arange(
                t[0], t[-1], 1 / interp_freq
            )  # Evenly resample at frequency
            if t[-1] > t[-1]:
                # Occasionally due to precision errors the last sample may be outside of range.
                t = t[:-1]

            # if fill_gaps:
            #     #  Find large gaps and forward fill @fixme This is inefficient
            #     (gaps,) = np.where(np.diff(self.tick_t) >= fill_gaps)
            #     for i in gaps:
            #         self.interpolator[(t >= self.tick_t[i]) & (t < self.tick_t[i + 1])] = (
            #             self.tick_pos[i]
            #         )

            interp_pos = cls._interpolator(interp_t)
            return interp_t, interp_pos
        else:
            print(
                "No interpolator set, do that first by calling the init_interpolator function"
            )

    @classmethod
    def get_movements(
        cls,
        t: np.ndarray,
        pos: np.ndarray,
        freq: float,
        pos_thresh=0.03,
        t_thresh=0.5,
        min_gap=0.1,
        pos_thresh_onset=1.5,
        min_dur=0.05,
    ) -> dict:
        """ Detect wheel movements. Uses interpolated positions

        Args:
            t (np.ndarray): Time values of the wheel trace
            pos (np.ndarray): Position values of the wheel trace
            freq (float): The sampling rate of the wheel data
            pos_thresh (float, optional): The minimum required movement during the t_thresh window to be considered part of a. Defaults to 0.03.
            t_thresh (float, optional): The time window over which to check whether the pos_thresh has been crossed. Defaults to 0.5.
            min_gap (float, optional): The minimum time between one movement's offset and another movement's onset in order to be
            considered separate.  Movements with a gap smaller than this are 'stictched together'. Defaults to 0.1.
            pos_thresh_onset (float, optional): A lower threshold for finding precise onset times.  The first position of each movement
            transition that is this much bigger than the starting position is considered the onset. Defaults to 1.5.
            min_dur (float, optional): The minimum duration of a valid movement.  Detected movements shorter than this are ignored. Defaults to 0.05.

        Returns:
            dict: Dictionary that has:
                onsets(np.ndarray): (N,2) array that has timestamps of detected movement onsets' indeces, and their values
                offsets(np.ndarray): (N,2) array that has timestamps of detected movement offsets' indeces, and their values
                peaks(np.ndarray) : (N,2) array that has peak positions' indeces, and their values
                speed_peaks(np.ndarray): (N,2) array that has peak speeds' indeces, and their values
        """
        # Wheel position must be evenly sampled
        movement_dict = {}
        dt = np.diff(t)
        assert np.all(np.abs(dt - dt.mean()) < 1e-10), "Values not evenly sampled"

        # Convert the time threshold into number of samples given the sampling frequency
        t_thresh_samps = int(np.round(t_thresh * freq))
        max_disp = np.empty(
            t.size, dtype=float
        )  # initialize array of total wheel displacement

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

        moving = (
            max_disp > pos_thresh
        )  # for each window is the change in position greater than our threshold?
        moving = np.insert(
            moving, 0, False
        )  # First sample should always be not moving to ensure we have an onset
        moving[-1] = False  # Likewise, ensure we always end on an offset

        onset_samps = np.where(~moving[:-1] & moving[1:])[0]
        offset_samps = np.where(moving[:-1] & ~moving[1:])[0]
        too_short = np.where((onset_samps[1:] - offset_samps[:-1]) / freq < min_gap)[0]
        for p in too_short:
            moving[offset_samps[p] : onset_samps[p + 1] + 1] = True

        onset_samps = np.where(~moving[:-1] & moving[1:])[0]
        onsets_disp_arr = np.empty((onset_samps.size, t_thresh_samps))
        c = 0
        cwt = 0
        while onset_samps.size != 0:
            i2proc = np.arange(BATCH_SIZE) + c
            icomm = np.intersect1d(
                i2proc[: -t_thresh_samps - 1], onset_samps, assume_unique=True
            )
            itpltz = np.intersect1d(
                i2proc[: -t_thresh_samps - 1],
                onset_samps,
                return_indices=True,
                assume_unique=True,
            )[1]
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

        movement_dict["onsets"] = np.hstack(
            (onset_samps.reshape(-1, 1), onsets.reshape(-1, 1))
        )
        movement_dict["offsets"] = np.hstack(
            (offset_samps.reshape(-1, 1), offsets.reshape(-1, 1))
        )

        # Calculate the peak amplitudes -
        # the maximum absolute value of the difference from the onset position
        peaks = np.array(
            [
                pos[m + np.abs(pos[m:n] - pos[m]).argmax()] - pos[m]
                for m, n in zip(onset_samps, offset_samps)
            ]
        )
        peak_samps = np.array(
            [
                m + np.abs(pos[m:n] - pos[m]).argmax()
                for m, n in zip(onset_samps, offset_samps)
            ]
        )
        peaks = np.array([pos[_i] - pos[_o] for _i,_o in zip(peak_samps,onset_samps)])
        # peak_amps = np.fromiter(peaks, dtype=float, count=onsets.size)
        
        movement_dict["peaks"] = np.hstack(
            (peak_samps.reshape(-1, 1), peaks.reshape(-1, 1))
        )

        N = 10  # Number of points in the Gaussian
        STDEV = 1.8  # Equivalent to a width factor (alpha value) of 2.5
        gauss = scipy.signal.windows.gaussian(
            N, STDEV
        )  # A 10-point Gaussian window of a given s.d.
        vel = scipy.signal.convolve(np.diff(np.insert(pos, 0, 0)), gauss, mode="same")
        vel = cls.get_filtered_velocity(pos,interp_freq=freq)

        # For each movement period, find the timestamp where the absolute velocity was greatest
        speed_peak_samps = np.array(
            [m + np.abs(vel[m:n]).argmax() for m, n in zip(onset_samps, offset_samps)]
        )
        speed_peaks = np.array([np.abs(vel[_v]) for _v in speed_peak_samps])
        speed_peaks = np.fromiter(speed_peaks, dtype=float, count=onsets.size)
        movement_dict["speed_peaks"] = np.hstack(
            (speed_peak_samps.reshape(-1, 1), speed_peaks.reshape(-1, 1))
        )

        return movement_dict

    @classmethod
    def get_filtered_velocity(
        cls,
        pos: np.ndarray,
        interp_freq: float = 5,
        corner_frequency: float = 2,
        order: int = 8,
    ) -> np.ndarray:
        """Compute wheel velocity from uniformly sampled wheel data.
        
        Args:
            pos (np.ndarray): Position values of the wheel trace
            interp_freq (float, optional): Interpolation frequency. Defaults to 5.
            corner_frequency (float, optional): Corner frequency of the filter. Defaults to 2.
            order (int, optional): Order of the filter. Defaults to 8.

        Returns:
            np.ndarray: Filtered velocity values
        """
        _wn = corner_frequency / interp_freq * 2
        sos = scipy.signal.butter(N=order, Wn=_wn, btype="lowpass", output="sos")

        velo = (
            np.insert(
                np.diff(scipy.signal.sosfiltfilt(sos, pos, padlen=len(pos) - 1)),
                0,
                0,
            )
            * interp_freq
        )
        return velo

    @classmethod
    def ticks_to_cm(cls, positions: np.ndarray) -> np.ndarray:
        """Convert wheel position samples to cm linear displacement

        Args:
            positions (np.ndarray): Position values of the wheel trace

        Returns:
            np.ndarray: Array with ticks converted to cm values
        """
        return positions / WHEEL_TICKS_PER_REV * np.pi * WHEEL_DIAMETER

    @classmethod
    def cm_to_rad(cls, positions: np.ndarray) -> np.ndarray:
        """Convert wheel position to radians.  This may be useful for e.g. calculating angular velocity

        Args:
            positions (np.ndarray): Position values of the wheel trace

        Returns:
            np.ndarray:  Array with cm values converted to radian values
        """
        return positions * (2 / WHEEL_DIAMETER)

    @classmethod
    def cm_to_deg(cls, positions: np.ndarray) -> np.ndarray:
        """Convert wheel position to degrees turned.  This may be useful for e.g. calculating velocity
        in revolutions per second

        Args:
            positions (np.ndarray): Position values of the wheel trace

        Returns:
            np.ndarray: Array with cm values converted to degree values
        """
        return positions / (WHEEL_DIAMETER * np.pi) * 360
