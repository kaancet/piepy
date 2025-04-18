import numpy as np
import polars as pl
from .io import display
from scipy.interpolate import interp1d


def add_total_iStim(rawdata: dict) -> dict:
    """Adds another column to the DataFrame where iStim increments for each presentation

    Args:
        rawdata (dict): The dictionary that has all the session data

    Returns:
        dict: rawdata with total_iStim added to vstim colums
    """
    display("Adding total iStim column")

    # if either iStim  or iTrial are not in the columns, make dummy zero arrays
    if "iStim" in rawdata["vstim"].columns:
        _iStim = rawdata["vstim"]["iStim"]
    else:
        _iStim = pl.Series("iStim", np.zeros(len(rawdata["vstim"])))

    if "iTrial" in rawdata["vstim"].columns:
        _iTrial = rawdata["vstim"]["iTrial"]
    else:
        _iTrial = pl.Series("iTrial", np.zeros(len(rawdata["vstim"])))

    # check if either itrial or istim is monotonically increasing, if so take the one that i
    _iStim_monot = np.diff(_iStim.drop_nulls().unique(maintain_order=True).to_list())
    _iTrial_monot = np.diff(_iTrial.drop_nulls().unique(maintain_order=True).to_list())
    if len(_iStim_monot) and np.all(_iStim_monot > 0):
        # istim monotonic, take that (scenario 1)
        rawdata["vstim"] = rawdata["vstim"].with_columns(
            pl.when(pl.col("iTrial").first() == 0)
            .then(pl.col("iStim") + 1)
            .otherwise(pl.col("iStim"))
            .alias("total_iStim")
        )
    elif len(_iTrial_monot) and np.all(_iTrial_monot > 0):
        # itrial is monotonic, take that (scenario 3)
        rawdata["vstim"] = rawdata["vstim"].with_columns(
            pl.when(pl.col("iTrial").first() == 0)
            .then((pl.col("iTrial") + 1))
            .otherwise(pl.col("iTrial"))
            .alias("total_iStim")
        )
    else:
        # for every unique iTrial, run on istims
        incr = []
        _observed_iTrials = []
        ctr = 1
        for t, s in zip(_iTrial.to_list(), _iStim.to_list()):
            if s is None or t is None:
                incr.append(None)
            else:
                if t not in _observed_iTrials:
                    _observed_iStims = []
                    if s not in _observed_iStims:
                        _observed_iStims.append(s)
                        ctr += 1
                incr.append(ctr)
        total_stim = pl.Series("total_iStim", incr)
        rawdata["vstim"] = rawdata["vstim"].with_columns(total_stim)

    return rawdata


def compare_cam_logging(rawdata: dict) -> dict:
    """Compares the camlogs with corresponding riglog recordings

    Args:
        rawdata (dict): The dictionary that has all the session data

    Returns:
        dict: rawdata with camlog lengths fixed
    """
    # !! IMPORTANT !!
    # rawdata keys that end only with 'cam' are from the rig
    # the ones that end with 'cam_log' are from labcams
    rig_cams = [k for k in rawdata.keys() if k.endswith("cam")]
    labcam_cams = [k for k in rawdata.keys() if k.endswith("cam_log")]

    assert len(rig_cams) == len(labcam_cams), (
        f"Number of camlogs in rig({len(rig_cams)}) and labcams({len(labcam_cams)}) are not equal!! "
    )

    for i, lab_cam_key in enumerate(labcam_cams):
        rig_cam_frames = len(rawdata[rig_cams[i]])
        labcams_frames = len(rawdata[lab_cam_key])

        if labcams_frames < rig_cam_frames:
            display(
                f"{rig_cam_frames - labcams_frames} logged frame(s) not recorded by {lab_cam_key}!!",
                color="yellow",
            )
        elif labcams_frames > rig_cam_frames:
            # remove labcams camlog frames if they are more than riglog recordings
            display(
                f"{labcams_frames - rig_cam_frames} logged frame(s) not logged in {rig_cams[i]}!",
                color="yellow",
            )

            rawdata[lab_cam_key] = rawdata[lab_cam_key].slice(
                0, rig_cam_frames
            )  # removing extra recorded frames
            if len(rawdata[lab_cam_key]) == rig_cam_frames:
                display("Camlogs are equal now!", color="cyan")

    return rawdata


def extract_trial_count(rawdata: dict) -> dict:
    """Extracts the trial no from state changes, this works for stimpy for now

    Args:
        rawdata (dict): The dictionary that has all the session data

    Returns:
        dict: rawdata with state machine trial no increments fixed
    """
    if len(rawdata["statemachine"]["trialNo"].unique()) == 1:
        # if no trial change logged iin state data
        display(
            "State machine trial increment faulty, extracting from state changes...",
            color="yellow",
        )

        trialends = rawdata["statemachine"].with_columns(
            pl.when(pl.col("transition").str.contains("trialend"))
            .then(1)
            .otherwise(0)
            .alias("end_flag")
        )

        trial_no = []
        t_cntr = 1
        for i in trialends["end_flag"].to_list():
            trial_no.append(t_cntr)
            if i:
                t_cntr += 1

        new_trial_no = pl.Series("trialNo", trial_no)
        rawdata["statemachine"] = rawdata["statemachine"].with_columns(new_trial_no)

    return rawdata


def stitch_logs(data_list: list, is_stimlog: bool) -> dict:
    """Stitches the seperate log files

    Args:
        data_list (list): List of dataframes to be stitched
        is_stimlo (bool): Flag to indicate the data is stimlog or not
    """

    final_data = data_list[0]
    for i in range(len(data_list) - 1):
        to_append = data_list[i + 1]  # skipping the first
        for k, v in final_data.items():
            if is_stimlog:
                # stimlog
                if k == "vstim":
                    to_append[k] = to_append[k].with_columns(
                        [
                            (pl.col("presentTime") + v[-1, "presentTime"]),
                            (pl.col("iTrial") + v[-1, "iTrial"] + 1),
                        ]
                    )
                elif k == "statemachine":
                    to_append[k] = to_append[k].with_columns(
                        [
                            (pl.col("elapsed") + v[-1, "elapsed"]),
                            (pl.col("cycle") + v[-1, "cycle"]),
                        ]
                    )
            else:
                # riglog
                if not to_append[k].is_empty():
                    if k not in ["reward", "position", "screen"]:
                        # adjust the values
                        to_append[k] = to_append[k].with_columns(
                            pl.col("value") + v[-1, "value"] + 1
                        )
                    elif k == "screen":
                        to_append[k] = to_append[k].slice(1)
                        to_append[k] = to_append[k].with_columns(
                            pl.col("value") + v[-1, "value"]
                        )

                    to_append[k] = to_append[k].with_columns(
                        [
                            (pl.col("timereceived") + v[-1, "timereceived"]),
                            (pl.col("duinotime") + v[-1, "duinotime"]),
                        ]
                    )
            final_data[k] = pl.concat([v, to_append[k]])
    return final_data


def extrapolate_time(rawdata: dict) -> dict:
    """Extrapolates duinotime from screen indicator

    Args:
        rawdata (dict): The dictionary that has all the session data

    Returns:
        dict: rawdata
    """
    if "vstim" in rawdata.keys() and "screen" in rawdata.keys():
        indkey = "not found"
        fliploc = []
        if "indicatorFlag" in rawdata["vstim"].columns:
            indkey = "indicatorFlag"
            fliploc = np.where(
                np.diff(np.hstack([0, rawdata["vstim"]["indicatorFlag"], 0])) != 0
            )[0]
        elif "photo" in rawdata["vstim"].columns:
            indkey = "photo"
            vstim_data = rawdata["vstim"].to_pandas()
            fliploc = np.where(
                np.diff(np.hstack([0, vstim_data["photo"] == 0, 0])) != 0
            )[0]

        if len(rawdata["screen"]) == len(fliploc):
            temp = interp1d(
                fliploc, rawdata["screen"]["duinotime"], fill_value="extrapolate"
            )(np.arange(len(rawdata["vstim"]))).tolist()
            temp_df = pl.Series("duinotime", temp)
            rawdata["vstim"] = rawdata["vstim"].hstack([temp_df])
        else:
            display(
                "The number of screen pulses {0} does not match the visual stimulation {1}:{2} log.".format(
                    len(rawdata["screen"]), indkey, len(fliploc)
                ),
                color="yellow",
            )
    return rawdata


def fix_first_line_state_logging(rawdata: dict) -> dict:
    """

    Args:
        rawdata (dict): The dictionary that has all the session data

    Returns:
        dict: Rawdata with statemachine erronous offset fixed
    """
    if (
        rawdata["statemachine"]["elapsed"][0]
        > rawdata["vstim"]["presentTime"][0] * 1000
    ):
        # subtract the first log time from all statemachine elapsed
        # rawdata["statemachine"] = rawdata["statemachine"].with_columns(
        #     pl.col("elapsed") - rawdata["statemachine"]["elapsed"][0]
        # )
        # add the firstlog time to all vstim presentTime
        rawdata["vstim"] = rawdata["vstim"].with_columns(
            pl.col("presentTime") + rawdata["statemachine"]["elapsed"][0] / 1000
        )

        # reset the stateElapsed of the first statemachine log entry
        rawdata["statemachine"][0, "stateElapsed"] = 0

    return rawdata
