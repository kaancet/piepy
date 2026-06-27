from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import scipy.signal
from scipy.linalg import hankel
from scipy.interpolate import PchipInterpolator

WHEEL_RADIUS_CM = 3.1
WHEEL_TICKS_PER_REV = 1024

_MOVEMENTS = ("onsets", "offsets", "peaks", "speed_peaks")


class WheelTrace:
    def __init__(self, t: ArrayLike | None = None, pos: ArrayLike | None = None) -> None:
        self._interp: PchipInterpolator | None = None
        self.load(t, pos)

    def load(self, t: ArrayLike | None = None, pos: ArrayLike | None = None) -> "WheelTrace":
        """Bind a new trace to this instance and clear its interpolator; returns ``self``.

        Lets one ``WheelTrace`` be reused across trials (``wt.load(t, pos).process(...)``) instead of
        allocating a new object each time.
        """
        t = np.asarray([] if t is None else t, dtype=float)
        pos = np.asarray([] if pos is None else pos, dtype=float)
        if t.size != pos.size:
            raise ValueError(f"wheel t and pos differ in length ({t.size} vs {pos.size}).")
        if t.size:
            t, pos = self.fix_trace_timing(t, pos)
        self.t, self.pos = t, pos
        self._interp = None
        return self

    def process(self, reset_time: float, *, freq: float = 5, units: str = "rad", **movement_kw) -> dict:
        """Full pipeline: reset+interpolate -> convert units -> velocity -> movements.

        Returns a dict with ``t``, ``pos`` (in ``units``), ``tick``, ``reset_t``, ``reset_tick``,
        ``velocity``, ``movements`` and ``freq``. Total over signal variety (empty -> empty result).
        """
        reset_t, reset_tick, t_interp, tick_interp = self.reset_and_interpolate(reset_time, freq)
        pos = self._to_units(tick_interp, units)
        if pos.size:
            vel = self.velocity(pos, freq)
            mov = self.get_movements(t_interp, pos, freq, **movement_kw)
        else:
            vel = pos
            mov = {k: np.empty((0, 2)) for k in _MOVEMENTS}
        return {
            "t": t_interp,
            "pos": pos,
            "tick": tick_interp,
            "reset_t": reset_t,
            "reset_tick": reset_tick,
            "velocity": vel,
            "movements": mov,
            "freq": freq,
        }

    @staticmethod
    def fix_trace_timing(t: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Drop samples that break strict monotonicity of ``t`` (rig occasionally logs out of order)."""
        while t.size > 1 and not np.all(np.diff(t) > 0):
            if np.diff(t)[-1] < 0:  # last element is the usual culprit
                t, pos = t[:-1], pos[:-1]
            else:
                bad = np.where(np.diff(t) <= 0)[0]
                t, pos = np.delete(t, bad), np.delete(pos, bad)
        return t, pos

    @staticmethod
    def ticks_to_cm(positions: ArrayLike) -> np.ndarray:
        """Encoder ticks -> cm of linear surface displacement."""
        return np.asarray(positions) / WHEEL_TICKS_PER_REV * (2 * np.pi * WHEEL_RADIUS_CM)

    @staticmethod
    def cm_to_rad(positions: ArrayLike) -> np.ndarray:
        """cm of surface displacement -> radians turned."""
        return np.asarray(positions) / WHEEL_RADIUS_CM

    def _to_units(self, ticks: np.ndarray, units: str) -> np.ndarray:
        if units == "tick":
            return np.asarray(ticks, dtype=float)
        cm = self.ticks_to_cm(ticks)
        if units == "cm":
            return cm
        if units == "rad":
            return self.cm_to_rad(cm)
        raise ValueError(f"units must be 'tick', 'cm' or 'rad', got {units!r}.")

    @staticmethod
    def find_nearest(arr: ArrayLike, value: float) -> int:
        """Returns the index of the nearest

        Args:
            arr (ArrayLike): Array of values to search in
            value (float): Find the nearest value to this one that exists in the input arr

        Returns:
            int: index of the nearest value in the arr
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return np.nanargmin(np.abs(arr - value))

    def _build_interp(self, t: np.ndarray, pos: np.ndarray) -> None:
        """(Re)build the position interpolator. Robust to empty / single-sample traces."""
        if t.size == 0:
            self._interp = None
            return
        if t.size == 1:  # no movement: duplicate the point so Pchip has two
            t = np.append(t, t[0] + 10)
            pos = np.append(pos, pos[0])
        self._interp = PchipInterpolator(t, pos, extrapolate=True)

    def interpolate_trace(self, t: np.ndarray, interp_freq: float = 5) -> tuple[np.ndarray, np.ndarray]:
        """Evenly resample the built interpolator at ``interp_freq`` Hz over ``t``'s span."""
        if self._interp is None:
            raise ValueError("No interpolator; call reset_and_interpolate first.")
        if t.size < 2:
            return t, self._interp(t)
        interp_t = np.arange(t[0], t[-1], 1.0 / interp_freq)
        return interp_t, self._interp(interp_t)

    def reset_and_interpolate(
        self, reset_time: float, interp_freq: float = 5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Zero time at ``reset_time`` and position at t=0, then evenly resample.

        Returns ``(reset_t, reset_tick, t_interp, tick_interp)``. Empty trace -> four empty arrays.
        """
        t, pos = self.t, self.pos
        if t.size == 0:
            empty = np.array([])
            return empty, empty, empty, empty

        if not reset_time > t[0]:
            # first movement logged after the reset point: pad backwards holding the first position
            add_t = np.arange(reset_time, t[0], 50)
            t = np.append(add_t, t)
            pos = np.append(np.full(add_t.size, pos[0]), pos)

        reset_t = t - reset_time
        # position at t=0 via a single linear lookup -> offset so position is 0 there (ticks are ints)
        pos_at0 = np.interp(0.0, reset_t, pos)
        reset_tick = np.round(pos - pos_at0).astype(int)

        self._build_interp(reset_t, reset_tick)
        t_interp, tick_interp = self.interpolate_trace(reset_t, interp_freq)
        return reset_t, reset_tick, t_interp, tick_interp

    @staticmethod
    def velocity(pos: ArrayLike, freq: float, *, window_s: float = 0.05, polyorder: int = 3) -> np.ndarray:
        """Velocity (pos-units per second) of an evenly-sampled position trace.

        Savitzky-Golay derivative: fits a local polynomial of ``polyorder`` over a ``window_s``-second
        window and returns its first derivative. No temporal averaging, zero phase lag. ``window_s``
        is in seconds so it is independent of ``freq``. Falls back to a finite difference when the
        trace is too short for the fit.
        """
        pos = np.asarray(pos, dtype=float)
        n = pos.size
        if n < 2:
            return np.zeros(n)
        win = int(round(window_s * freq)) | 1  # odd
        win = min(win, n if n % 2 else n - 1)  # <= n, keep odd
        if win <= polyorder:  # too short to fit the polynomial -> plain finite difference
            return np.gradient(pos, 1.0 / freq)
        return scipy.signal.savgol_filter(pos, win, polyorder, deriv=1, delta=1.0 / freq)

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
        """Detect wheel movements. Uses interpolated positions

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
        movement_dict = {k: np.empty((0, 2)) for k in _MOVEMENTS}
        t = np.asarray(t, dtype=float)
        pos = np.asarray(pos, dtype=float)
        if t.size < 2:
            return movement_dict
        dt = np.diff(t)
        assert np.all(np.abs(dt - dt.mean()) < 1e-10), "Values not evenly sampled"

        # Convert the time threshold into number of samples given the sampling frequency
        t_thresh_samps = int(np.round(t_thresh * freq))
        if t.size <= t_thresh_samps or np.ptp(pos) == 0:  # too short / no movement
            return movement_dict

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
            moving[offset_samps[p] : onset_samps[p + 1] + 1] = True

        onset_samps = np.where(~moving[:-1] & moving[1:])[0]
        onsets_disp_arr = np.empty((onset_samps.size, t_thresh_samps))
        c = 0
        cwt = 0
        while onset_samps.size != 0:
            i2proc = np.arange(BATCH_SIZE) + c
            icomm = np.intersect1d(i2proc[: -t_thresh_samps - 1], onset_samps, assume_unique=True)
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

        if onset_samps.size == 0:  # filtering removed every candidate -> no movements
            return movement_dict

        movement_dict["onsets"] = np.hstack((onset_samps.reshape(-1, 1), onsets.reshape(-1, 1)))
        movement_dict["offsets"] = np.hstack((offset_samps.reshape(-1, 1), offsets.reshape(-1, 1)))

        # Calculate the peak amplitudes -
        # the maximum absolute value of the difference from the onset position
        peaks = np.array(
            [pos[m + np.abs(pos[m:n] - pos[m]).argmax()] - pos[m] for m, n in zip(onset_samps, offset_samps)]
        )
        peak_samps = np.array([m + np.abs(pos[m:n] - pos[m]).argmax() for m, n in zip(onset_samps, offset_samps)])
        peaks = np.array([pos[_i] - pos[_o] for _i, _o in zip(peak_samps, onset_samps)])
        # peak_amps = np.fromiter(peaks, dtype=float, count=onsets.size)

        movement_dict["peaks"] = np.hstack((peak_samps.reshape(-1, 1), peaks.reshape(-1, 1)))

        # peak speed within each movement (SavGol velocity -- no temporal averaging)
        vel = WheelTrace.velocity(pos, freq)

        # For each movement period, find the timestamp where the absolute velocity was greatest
        speed_peak_samps = np.array([m + np.abs(vel[m:n]).argmax() for m, n in zip(onset_samps, offset_samps)])
        speed_peaks = np.array([np.abs(vel[_v]) for _v in speed_peak_samps])
        speed_peaks = np.fromiter(speed_peaks, dtype=float, count=onsets.size)
        movement_dict["speed_peaks"] = np.hstack((speed_peak_samps.reshape(-1, 1), speed_peaks.reshape(-1, 1)))

        return movement_dict


@dataclass
class RTResult:
    """The movement matched to a response time, and how it was matched.

    ``source`` records provenance so downstream analyses can filter: ``"contain"`` (response fell
    inside a movement -- the clean case), ``"gap"`` (response landed just after a movement's offset,
    within ``gap_tol``), or ``"none"`` (no match). ``anticipatory`` flags a too-early onset
    (``< min_rt``, including pre-stimulus negative onsets) -- the RT is still returned, but tagged so
    it can be excluded from reaction-time conclusions.
    """

    reaction_time: float | None
    peak_speed: float | None
    source: str
    anticipatory: bool


def match_response_movement(
    movements: dict,
    resp_time: float | None,
    *,
    gap_tol: float = 100.0,
    min_rt: float = 150.0,
) -> RTResult:
    """Pick the movement that produced the response at ``resp_time`` -> its onset is the reaction time.

    Two passes (the order matters -- a *containing* movement must win over an earlier movement that
    merely ends just before ``resp_time``):

    1. **contain**: the movement with ``onset <= resp_time < offset``.
    2. **gap**: else the movement whose offset is *closest* to ``resp_time`` and within ``gap_tol``
       before it (detection thresholds occasionally clip the answering movement short).

    Returns the matched onset as ``reaction_time`` (with its ``peak_speed``), or a ``"none"`` result
    when ``resp_time`` is None / there are no movements / nothing matches. Onsets below ``min_rt``
    (or negative, i.e. pre-stimulus) are flagged ``anticipatory`` rather than dropped.
    """
    onsets = movements.get("onsets") if movements else None
    if resp_time is None or onsets is None or len(onsets) == 0:
        return RTResult(None, None, "none", False)

    on = onsets[:, 1]
    off = movements["offsets"][:, 1]
    spd = movements["speed_peaks"][:, 1]

    contain = np.where((on <= resp_time) & (resp_time < off))[0]
    if contain.size:
        i = int(contain[0])
        return RTResult(float(on[i]), float(spd[i]), "contain", bool(on[i] < min_rt))

    gaps = resp_time - off  # >=0 means resp is after this offset
    eligible = np.where((gaps >= 0) & (gaps <= gap_tol))[0]
    if eligible.size:
        i = int(eligible[np.argmin(gaps[eligible])])  # the closest preceding offset
        return RTResult(float(on[i]), float(spd[i]), "gap", bool(on[i] < min_rt))

    return RTResult(None, None, "none", False)
