"""Region-based quantification for 1P widefield movies.

Draw or load polygon regions, mask an (N, H, W) movie to each region, and
optionally reduce to a per-region statistic.

Coordinate convention: polygon vertices are stored as (x, y) == (col, row),
matching matplotlib display coords. The (row, col) swap needed by
skimage.draw.polygon2mask is handled once, inside regions_to_masks.
"""

import json

import numpy as np
import tifffile as tf
from skimage.draw import polygon2mask

_STATS = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "max": np.nanmax,
    "min": np.nanmin,
    "std": np.nanstd,
}


def _squeeze_movie(movie: np.ndarray) -> np.ndarray:
    """(N, 1, H, W) -> (N, H, W); (N, H, W) passes through."""
    movie = np.asarray(movie)
    if movie.ndim == 4 and movie.shape[1] == 1:
        return movie[:, 0, :, :]
    if movie.ndim == 3:
        return movie
    raise ValueError(f"movie must be (N,H,W) or (N,1,H,W), got shape {movie.shape}")


def reference_frame(movie: np.ndarray, source=None) -> np.ndarray:
    """Backdrop for drawing. Default: nan-mean projection over frames.

    source: None -> mean projection; str/Path -> tif on disk; ndarray -> used as-is.
    """
    if source is None:
        return np.nanmean(_squeeze_movie(movie), axis=0)
    if isinstance(source, np.ndarray):
        return source
    return tf.imread(str(source))


class RegionDrawer:
    """Interactive polygon drawer for notebooks (VSCode/Jupyter).

    Opens a persistent window with a PolygonSelector plus a name box and an
    "Add region" button. Draw a polygon (click vertices, click the first point
    to close it), type a name in the box and press <enter> (or click the
    button) to store it; the selector resets for the next region. Read the
    result from `.regions` -> {name: (V,2) float (x,y) array}.

        d = RegionDrawer(ref)   # window opens; draw, name, <enter>, repeat
        ...
        regions = d.regions

    ponytail: needs a native-window backend. macOS: `%matplotlib osx`. Inline
    backends (ipympl/widget) and Agg cannot drive the widgets.
    """

    def __init__(self, ref_img: np.ndarray):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import PolygonSelector, Button, TextBox

        self._PolygonSelector = PolygonSelector
        self.regions = {}
        self._cur = None

        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.18)
        self.ax.imshow(ref_img, cmap="gray")
        self.ax.set_title("draw a polygon, type a name, press <enter>")

        # name box + add button along the bottom
        self.textbox = TextBox(
            self.fig.add_axes([0.15, 0.05, 0.45, 0.06]), "name ", initial=""
        )
        self.button = Button(self.fig.add_axes([0.65, 0.05, 0.2, 0.06]), "Add region")
        self.textbox.on_submit(self._commit)  # <enter> in the box
        self.button.on_clicked(lambda _evt: self._commit(self.textbox.text))

        self._new_selector()
        plt.show()

    def _new_selector(self):
        self.selector = self._PolygonSelector(self.ax, self._onselect)

    def _onselect(self, pts):
        self._cur = np.asarray(pts, dtype=float)

    def _commit(self, name: str) -> None:
        name = (name or "").strip()
        if not name:
            self.ax.set_title("!! type a name first")
            self.fig.canvas.draw_idle()
            return
        if self._cur is None or len(self._cur) < 3:
            self.ax.set_title("!! draw and close a polygon first")
            self.fig.canvas.draw_idle()
            return
        self.save(name)
        self.textbox.set_val("")
        self.ax.set_title(f"added '{name}' - draw the next region")
        self.fig.canvas.draw_idle()

    def save(self, name: str) -> None:
        """Snapshot the current polygon under `name`, then reset for the next.

        Also callable directly from a cell if you prefer not to use the button.
        """
        if not name:
            raise ValueError("name must be non-empty")
        if self._cur is None or len(self._cur) < 3:
            raise ValueError("no completed polygon; draw one (close it) before saving")
        self.regions[name] = self._cur
        # draw a static outline so saved regions stay visible
        self.ax.fill(self._cur[:, 0], self._cur[:, 1], alpha=0.25, edgecolor="r")
        self.ax.text(
            self._cur[:, 0].mean(),
            self._cur[:, 1].mean(),
            name,
            color="r",
            ha="center",
            va="center",
        )
        self.selector.disconnect_events()
        self._cur = None
        self._new_selector()
        self.fig.canvas.draw_idle()


def draw_regions(ref_img: np.ndarray, names=None) -> dict:
    """Blocking hand-draw for a plain terminal IPython/python session only.

    Draw a polygon, close the window; type a name at the prompt (blank ends).
    Does NOT work in VSCode/Jupyter notebooks (the GUI loop makes
    `plt.show(block=True)` non-blocking) -> use `RegionDrawer` there.
    Returns {name: (V,2) float (x,y) array}.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    names = list(names) if names is not None else None
    regions = {}
    while True:
        fig, ax = plt.subplots()
        ax.imshow(ref_img, cmap="gray")
        ax.set_title("draw polygon, close window when done")

        verts = {}

        def _onselect(pts, _store=verts):
            _store["v"] = np.asarray(pts, dtype=float)

        selector = PolygonSelector(ax, _onselect)
        plt.show(block=True)  # blocks only in a real terminal event loop
        selector.disconnect_events()

        if "v" not in verts or len(verts["v"]) < 3:
            break  # no polygon drawn -> done

        if names is not None:
            if not names:
                break
            name = names.pop(0)
        else:
            name = input("region name (blank to stop): ").strip()
            if not name:
                break
        regions[name] = verts["v"]
    return regions


def save_regions(regions: dict, path: str) -> None:
    """Write {name: (V,2)} vertices to json as {name: [[x,y],...]}."""
    serializable = {k: np.asarray(v).tolist() for k, v in regions.items()}
    with open(path, "w") as fp:
        json.dump(serializable, fp, indent=2)


def load_regions(path: str) -> dict:
    """Read json written by save_regions -> {name: (V,2) float array}."""
    with open(path) as fp:
        raw = json.load(fp)
    return {k: np.asarray(v, dtype=float) for k, v in raw.items()}


def regions_to_masks(regions: dict, shape) -> dict:
    """{name: (V,2) (x,y) verts} -> {name: (H,W) bool mask}.

    shape is the (H, W) frame shape. Raises if regions overlap (masks must be
    mutually exclusive) or if a polygon produces an empty mask.
    """
    h, w = shape[-2], shape[-1]
    masks = {}
    for name, verts in regions.items():
        verts = np.asarray(verts, dtype=float)
        # (x,y) -> (row,col) for skimage
        verts_rc = verts[:, ::-1]
        mask = polygon2mask((h, w), verts_rc)
        if not mask.any():
            raise ValueError(
                f"region '{name}' produced an empty mask (out of frame bounds?)"
            )
        masks[name] = mask

    # mutual-exclusivity check
    names = list(masks)
    overlap_total = np.zeros((h, w), dtype=int)
    for m in masks.values():
        overlap_total += m
    if (overlap_total > 1).any():
        clashes = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n = int((masks[names[i]] & masks[names[j]]).sum())
                if n:
                    clashes.append(f"'{names[i]}' & '{names[j]}' ({n}px)")
        raise ValueError(
            "regions must be mutually exclusive; overlaps: " + ", ".join(clashes)
        )
    return masks


def apply_regions(movie: np.ndarray, regions: dict, fill=np.nan) -> dict:
    """Mask movie to each region. Returns {name: (N,H,W)} with pixels outside
    the region set to `fill`.

    `regions` may be vertices ({name:(V,2)}) or precomputed boolean masks
    ({name:(H,W) bool}).
    """
    movie = _squeeze_movie(movie).astype(float)
    n, h, w = movie.shape

    first = next(iter(regions.values())) if regions else None
    is_mask = isinstance(first, np.ndarray) and first.dtype == bool
    masks = regions if is_mask else regions_to_masks(regions, (h, w))

    out = {}
    for name, mask in masks.items():
        if mask.shape != (h, w):
            raise ValueError(f"mask '{name}' shape {mask.shape} != frame shape {(h, w)}")
        masked = np.where(mask[None, :, :], movie, fill)
        out[name] = masked
    return out


def quantify(masked_movies: dict, stat="mean", axis="spatial", **kwargs) -> dict:
    """Reduce each region's (N,H,W) masked movie to a statistic.

    axis="spatial" -> reduce over (H,W) -> (N,) per region.
    axis="frames"  -> reduce over N     -> (H,W) per region.
    stat: name in {mean,median,max,min,std} (nan-aware) OR a callable
          func(arr, axis=<int|tuple>, **kwargs).
    """
    if axis == "spatial":
        red_axis = (1, 2)
    elif axis == "frames":
        red_axis = 0
    else:
        raise ValueError(f"axis must be 'spatial' or 'frames', got {axis!r}")

    if callable(stat):
        func = stat
    elif stat in _STATS:
        func = _STATS[stat]
    else:
        raise ValueError(f"unknown stat {stat!r}; use {list(_STATS)} or a callable")

    return {
        name: func(mov, axis=red_axis, **kwargs) for name, mov in masked_movies.items()
    }


def _demo() -> None:
    """Self-check: masking, overlap error, both reduce axes."""
    n, h, w = 4, 10, 10
    movie = np.ones((n, h, w))
    movie *= np.arange(1, n + 1)[:, None, None]  # frame f == value f

    # square region covering rows/cols 2..5 -> vertices as (x,y)
    sq = np.array([[2, 2], [5, 2], [5, 5], [2, 5]], dtype=float)
    regions = {"a": sq}

    masked = apply_regions(movie, regions)
    m = masked["a"]
    assert m.shape == (n, h, w)
    assert np.isnan(m[0, 0, 0])  # outside region
    inside = m[1][~np.isnan(m[1])]
    assert np.all(inside == 2.0)  # frame 1 -> value 2

    ts = quantify(masked, "mean", axis="spatial")["a"]
    assert ts.shape == (n,) and np.allclose(ts, [1, 2, 3, 4])

    proj = quantify(masked, "max", axis="frames")["a"]
    assert proj.shape == (h, w)
    assert np.nanmax(proj) == 4.0

    # custom callable with kwarg passthrough
    proj_p = quantify(masked, np.nanpercentile, axis="frames", q=50)["a"]
    assert proj_p.shape == (h, w)

    # overlap must raise
    overlapping = {"a": sq, "b": np.array([[3, 3], [6, 3], [6, 6], [3, 6]], float)}
    try:
        regions_to_masks(overlapping, (h, w))
        raise AssertionError("expected overlap ValueError")
    except ValueError as e:
        assert "mutually exclusive" in str(e)

    print("regions._demo ok")


if __name__ == "__main__":
    _demo()
