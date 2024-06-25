from piepy.plotters.bokeh_plot.bokeh_base import *

from bokeh.models import LinearColorMapper, ColorBar
from bokeh.layouts import row
import numpy as np


class PatternImageGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds = None

    @staticmethod
    def map_range(img: np.ndarray) -> np.ndarray:
        return np.interp(img, [np.min(img), np.max(img)], [0, 65536])

    @staticmethod
    def make_palette_name(name: str) -> None:
        return f"{name}256"

    def set_palette(self, palette: str = "Greys256") -> None:
        if self.fig is not None:
            p_name = self.make_palette_name(palette)
            mapper = LinearColorMapper(palette=p_name, low=0, high=65536)
            self.img_glyph.glyph.color_mapper = mapper
            self.color_bar.color_mapper = mapper

    def set_colormap_range(self, low: int = 0, high: int = 65536) -> None:
        if self.fig is not None:
            self.img_glyph.glyph.color_mapper.update(low=low, high=high)
            self.color_bar.color_mapper.update(low=low, high=high)

    def reset_cds(self) -> None:
        self.cds = None

    def set_cds(self, data: np.ndarray) -> None:
        img = np.fliplr(data[::-1])
        img = self.map_range(img)
        temp = {"img": [img]}

        if self.cds is None:
            self.cds = ColumnDataSource(data=temp)
        else:
            self.cds.data = temp

    def plot(self) -> None:

        f = figure(width=610, height=450, toolbar_location="below")
        color_mapper = LinearColorMapper(palette="Greys256", low=0, high=65536)
        self.img_glyph = f.image(
            image="img", x=0, y=0, dw=5, dh=5, source=self.cds, color_mapper=color_mapper
        )

        self.color_bar = ColorBar(color_mapper=color_mapper)

        f.axis.visible = False
        f.add_layout(self.color_bar, place="right")

        self.fig = f
