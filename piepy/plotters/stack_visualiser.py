import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from matplotlib.widgets import Button, Slider
from matplotlib.gridspec import GridSpec

from piepy.imaging.onep.facecam.facecamAnalysis import FaceCamAnalysis
from piepy.imaging.onep.eyecam.eyecamAnalysis import EyeCamAnalysis


def play_stacks(stacks: np.ndarray, colormap: str = "grays") -> None:
    """stack shape -> (frames, width, height)"""
    frame_id = 0

    def get_frame(idx: int) -> np.ndarray:
        return stacks[idx, :, :]

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()

    img = ax.imshow(get_frame(frame_id), cmap=colormap)
    ax.set_axis_off()

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frequency.
    axframe = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=axframe,
        label="Frame #",
        valmin=0,
        valmax=stacks.shape[0],
        valinit=frame_id,
        valstep=1,
    )

    def update(val) -> None:
        img.set_data(get_frame(frame_slider.val))
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)
    plt.show()


class TrialVisualiser:
    def __init__(
        self, trial_data: pl.DataFrame, facecam: FaceCamAnalysis, eyecam: EyeCamAnalysis
    ) -> None:
        self.trial_data = trial_data
        self._face = facecam
        self._eye = eyecam

        self.get_trial_frames()

    def get_trial_frames(self) -> None:
        _face_ids = self.trial_data[0, "facecam_frame_ids"].to_list()
        self.facecam_frames = self._face.tif_stack[
            _face_ids[0] : _face_ids[1] + 1, 0, :, :
        ]

        _eye_ids = self.trial_data[0, "eyecam_frame_ids"].to_list()
        self.eyecam_frames = self._eye.tif_stack[_eye_ids[0] : _eye_ids[1] + 1, 0, :, :]

    def show(self, colormap: str = "gray") -> None:
        time_in_trial = 1

        def get_frame_with_time(cam_type: str, time: float) -> np.ndarray:
            """ """
            if cam_type == "face":
                frame_id = round(time / self._face.frame_t)
                frame = self.facecam_frames[frame_id, :, :]

            elif cam_type == "eye":
                frame_id = round(time / self._eye.frame_t)
                frame = self.eyecam_frames[frame_id, :, :]

            return frame

        fig = plt.figure(layout="constrained")

        gs = GridSpec(3, 2, figure=fig, height_ratios=[5, 5, 1])
        ax_facecam = fig.add_subplot(gs[0, 0])
        ax_eyecam = fig.add_subplot(gs[0, 1])
        ax_trial = fig.add_subplot(gs[1, :])
        ax_slider = fig.add_subplot(gs[2, :])

        # facecam
        # face_img = ax_facecam.imshow(get_frame_with_time("face",time_in_trial),cmap=colormap)
        # ax_facecam.set_axis_off()

        # #eyecam
        # eye_img = ax_eyecam.imshow(get_frame_with_time("eye",time_in_trial),cmap=colormap)
        # ax_eyecam.set_axis_off()

        # trial
        line = ax_trial.axvline(time_in_trial, color="r", linewidth=2)
        ax_trial.set_xlim([0, 2000])

        # Make a horizontal slider to control the frequency.
        time_slider = Slider(
            ax=ax_slider,
            label="Time in Trial (ms)",
            valmin=1,
            valmax=2000,
            valinit=time_in_trial,
            valstep=10,
        )

        def update(val) -> None:
            # face_frame = get_frame_with_time("face",val)
            # eye_frame = get_frame_with_time("eye",val)

            # face_img.set_data(face_frame)
            # eye_img.set_data(eye_frame)

            # line.set_xdata([val])

            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        plt.show()
