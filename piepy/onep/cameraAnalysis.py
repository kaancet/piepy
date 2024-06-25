from polars import DataFrame
from .stacks import *
from .myio import *
from .retinoutils import *
from ..utils import *


class CamDataAnalysis:
    def __init__(self, data: pl.DataFrame, runpath: str) -> None:
        self.data = data
        self.runpath = runpath

    def get_tif_stack(self) -> None:
        """Gets the stack as a np.memmap"""
        self.tif_stack = load_stack(self.runpath, nchannels=1)

    def get_camlogs(self) -> None:
        """Parses the camlog"""
        dir_content = os.listdir(self.runpath)

        # only get the camlog and not the tiffs
        cam_log = [f for f in dir_content if f.endswith("log")]

        if len(cam_log) == 1:
            p_camlog = pjoin(self.runpath, cam_log[0])
            self.camlog, self.camlog_comment, _ = parseCamLog(p_camlog)
        elif len(cam_log) > 1:
            raise IOError(f"Multiple camlogs present in run directory {self.runpath}")
        elif len(cam_log) == 0:
            raise IOError(f"!!No camlogs present in run directory!! {self.runpath}")

    def get_frame_time(self) -> None:
        """Gets the avg frame time from experiment duration and frame count"""
        avg_ftime = np.mean(np.diff(self.camlog["timestamp"]))
        if avg_ftime == 0:
            tmp = [i for i in self.camlog_comment if "# [" in i]
            exp_start = dt.strptime(tmp[0].split("]")[0][-8:], "%H:%M:%S")
            exp_end = dt.strptime(tmp[-1].split("]")[0][-8:], "%H:%M:%S")

            exp_dur = exp_end - exp_start
            exp_dur = exp_dur.seconds

            total_frame_count = len(self.camlog)

            frame_t = exp_dur / total_frame_count  # in seconds
        else:
            # potential failsafe for pycams rig differences?
            if avg_ftime < 1:
                frame_t = avg_ftime
            else:
                frame_t = (
                    avg_ftime / 10_000
                )  # pycams measures in 10 microsecond intervals
        self.frame_t = frame_t * 1000
        display(f"Avg. frame time: {self.frame_t} ms")

    def set_minimum_dur(self, duration: float = None) -> None:
        """Sets all the rows in the frame mat to have the same amount of frames
        If no value provided, uses the trial with minimum number of frames"""
        if duration is None:
            frame_count = np.min(self.frame_mat[:, -1])
        else:
            frame_count = int(np.round(duration / self.frame_t))
            if frame_count < 0:
                raise ValueError(
                    f'Fixed frame count can"t be negative. got {frame_count}'
                )

        # getting rid of trials with less then minimum frame (for recordings during the task)
        mask = np.ones((len(self.frame_mat)), dtype="bool")
        mask[np.where(self.frame_mat[:, -1] < frame_count)] = False
        self.frame_mat = self.frame_mat[mask, :]

        # fix min frames
        for i in range(self.frame_mat.shape[0]):
            if self.frame_mat[i, -1] > frame_count:
                frame_diff = self.frame_mat[i, -1] - frame_count
                self.frame_mat[i, -1] = frame_count
                # remove frames from the end
                self.frame_mat[i, -2] -= frame_diff

    def scale_camlog_timeunit(self, scale_by: float) -> None:
        """Scales the timestamp by desired value"""
        self.camlog = self.camlog.with_columns(
            (pl.col("timestamp") * scale_by).alias("timestamp")
        )
