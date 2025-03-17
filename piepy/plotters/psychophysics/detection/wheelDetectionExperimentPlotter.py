import polars as pl

# constant animal colors
ANIMAL_COLORS = {
    "KC139": "#332288",
    "KC141": "#117733",
    "KC142": "#DDCC77",
    "KC143": "#AA4499",
    "KC144": "#882255",
    "KC145": "#88CCEE",
    "KC146": "#275D6D",
    "KC147": "#F57A6C",
    "KC148": "#ADFA9A",
    "KC149": "#A45414",
    "KC150": "#0000FF",
    "KC151": "#00FF11",
    "KC152": "#FFAA33",
}

# values from allen ccf_2022, using the shoelace algorithm to calculate the area from area boundary coordinates
AREA_SIZE = {
    "V1": 4.002,
    "HVA": 2.925,
    "dorsal": 1.428,
    "ventralPM": 1.496,
    "LM": 0.571,
    "AL": 0.389,
    "RL": 0.583,
    "PM": 0.719,
    "AM": 0.456,
}
# "LI":0.207
# "cortex" : 6.927,

# hierarchy scores from steinmetz?
AREA_SCORE = {}


def group_data(
    self, data: pl.DataFrame, group_by: list, sort_by_group: bool = True
) -> pl.DataFrame:
    """ """

    # check if group_by names are in the dataframe columns
    for c in group_by:
        if c not in data.columns:
            raise ValueError(
                f"{c} not a valid column name in given DataFrame, try one of: {data.columns}"
            )

    q = (
        data.lazy()
        .group_by(group_by)
        .agg(
            [
                (pl.col("outcome") != 1).sum().alias("trial_count"),
                (pl.col("outcome") == 1).sum().alias("correct_count"),
                (pl.col("outcome") == 0).sum().alias("miss_count"),
                (
                    pl.when(pl.col("outcome") == 1)
                    .then(pl.col("reaction_times"))
                    .alias("_temp_reaction_times")
                ),
                (pl.col("session_no").first()),
                (pl.col("stimkey").first()),
                (pl.col("stim_label").first()),
            ]
        )
        .drop_nulls()
    )

    if sort_by_group:
        q = q.sort(group_by)

    # reorder stim_label to last column
    cols = q.collect_schema().names()
    move_cols = ["stimkey", "stim_label"]
    for to_del in move_cols:
        _del = cols.index(to_del)
        del cols[_del]
    cols.extend(move_cols)
    q = q.select(cols)


class AllAreasScatterPlotter:
    def __init__(self, data, **kwargs) -> None:
        set_style(kwargs.pop("style", "presentation"))
        self.plot_data = data
        self.fig = None
        self.area_list = [
            "V1",
            "HVA",
            "dorsal",
            "ventralPM",
            "LM",
            "AL",
            "RL",
            "PM",
            "AM",
        ]
        self.add_area_size()

    def add_area_size(self) -> None:
        """Adds the size of the area as a column"""
        area_list = self.plot_data["area"].to_list()
        size_list = [AREA_SIZE.get(a) for a in area_list]
        self.plot_data = self.plot_data.with_columns(
            pl.Series(name="area_size", values=size_list)
        )

    @staticmethod
    def add_jitter(arr, jitter_lims: list = [-0.2, 0.2]) -> np.ndarray:
        """Adds jitter in x-dimension"""
        arr = np.array(arr)  # polars returns an immutable numpy array, this changes that

        jitter = np.random.choice(
            np.linspace(jitter_lims[0], jitter_lims[1], len(arr)), len(arr), replace=True
        )
        arr = arr + jitter
        return arr

    def plot_contrast_effect_difference(
        self, ax: plt.Axes = None, metric: str = "delta_HR", **kwargs
    ) -> plt.Axes:
        """ """
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        use_animal_colors = kwargs.pop("color_animals", False)

        analyzer = DetectionAnalysis()

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            idx = i + 1
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_sessions = area_df["session_no"].unique().to_list()
            area_contrast_hr_diff = []
            per_session_errs = []
            colors = []
            for sesh in uniq_sessions:
                sesh_df = area_df.filter(pl.col("session_no") == sesh)

                analyzer.set_data(sesh_df)

                d_hr = analyzer.get_deltahits()
                hit_rate_contra = d_hr.filter(pl.col("stim_side") == "contra")

                if len(hit_rate_contra):
                    hit_rate_contra = hit_rate_contra.sort("contrast")
                    _diff = (
                        hit_rate_contra[0, metric] - hit_rate_contra[1, metric]
                    )  # 12.5 - 50 contrast
                    _norm_diff = _diff / (
                        hit_rate_contra[0, metric] + hit_rate_contra[1, metric]
                    )
                    area_contrast_hr_diff.append(100 * _norm_diff)
                    per_session_errs.append(100 * hit_rate_contra[0, f"{metric}_err"])
                    if use_animal_colors:
                        colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                    else:
                        colors.append("#424242")

            _x = self.add_jitter([idx] * len(area_contrast_hr_diff))

            ax.scatter(
                _x, area_contrast_hr_diff, s=250, color=colors, alpha=1, linewidths=0
            )

            # do violin
            if kwargs.get("violin_plot", False):
                parts = ax.violinplot(
                    area_contrast_hr_diff,
                    [idx],
                    showmedians=False,
                    showextrema=False,
                    widths=0.5,
                    side="low" if j == 0 else "high",
                )

                for pc in parts["bodies"]:
                    pc.set_facecolor("#424242")
                    pc.set_alpha(0.6)

            quartile1, medians, quartile3 = np.percentile(
                area_contrast_hr_diff, [25, 50, 75]
            )
            ax.scatter(idx, medians, marker="_", color="k", s=100, zorder=3)
            ax.vlines(
                idx,
                quartile1,
                quartile3,
                color="k",
                linestyle="-",
                linewidth=1.5,
            )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel(f"{metric} difference 12.5% - 50%", fontsize=fontsize)
        ax.set_yticks([0, 25, 50, 75, 100])

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(1, len(_area_list) + 1))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        return ax

    def plot_hit_rates(
        self,
        ax: plt.Axes = None,
        metric: str = "percent_delta_HR",
        contrast: float = -1,
        **kwargs,
    ) -> plt.Axes:
        """metric can be delta_HR, percent_delta_HR, delta_HR_baseline"""

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        analyzer = DetectionAnalysis()

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_contrast = area_df["contrast"].drop_nulls().unique().sort().to_numpy()
            uniq_contrast = [c for c in uniq_contrast if c not in [100, 6.25, 0]]

            contrast_axis_shift = [-0.1, 0.1]
            for j, c in enumerate(uniq_contrast):

                # to plot single contrast or both
                if contrast == c:
                    contrast_ind = i
                elif contrast != -1 and contrast != c:
                    continue
                else:
                    contrast_ind = i + contrast_axis_shift[j]

                contrast_df = area_df.filter(
                    (pl.col("contrast") == c) | (pl.col("contrast") == 0)
                )

                uniq_sessions = area_df["session_no"].unique().to_list()
                area_hrs = []
                per_session_errs = []
                colors = []
                for sesh in uniq_sessions:
                    sesh_df = contrast_df.filter(pl.col("session_no") == sesh)

                    analyzer.set_data(sesh_df)

                    d_hr = analyzer.get_deltahits()
                    hit_rate_contra = d_hr.filter(pl.col("stim_side") == "contra")

                    if len(hit_rate_contra):
                        area_hrs.append(100 * hit_rate_contra[0, metric])
                        per_session_errs.append(100 * hit_rate_contra[0, f"{metric}_err"])
                        if kwargs.pop("color_animals", False):
                            colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                        else:
                            _clr = "#bfbfbf" if c == 12.5 else "#424242"
                            colors.append(_clr)

                _x = self.add_jitter([contrast_ind] * len(area_hrs))

                # ax.errorbar(_x, area_hrs,per_session_errs,
                #             marker='o',
                #             color = _clr,
                #             linewidth=0,
                #             elinewidth=plt.rcParams['lines.linewidth'])

                ax.scatter(_x, area_hrs, s=250, color=colors, alpha=1, linewidths=0)

                # do violin
                if kwargs.get("violin_plot", False):
                    parts = ax.violinplot(
                        area_hrs,
                        [contrast_ind],
                        showmedians=False,
                        showextrema=False,
                        widths=0.5,
                        side="low" if j == 0 else "high",
                    )

                    for pc in parts["bodies"]:
                        if j == 0:
                            pc.set_facecolor("#bfbfbf")
                        else:
                            pc.set_facecolor("#424242")
                        pc.set_alpha(0.6)

                quartile1, medians, quartile3 = np.percentile(area_hrs, [25, 50, 75])
                ax.scatter(contrast_ind, medians, marker="_", color="k", s=100, zorder=3)
                ax.vlines(
                    contrast_ind,
                    quartile1,
                    quartile3,
                    color="k",
                    linestyle="-",
                    linewidth=1.5,
                )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel(metric, fontsize=fontsize)
        ax.set_yticks([0, 25, 50, 75, 100])

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list)))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        return ax

    def plot_resp_times(
        self, ax: plt.Axes = None, contrast: float = -1, **kwargs
    ) -> plt.Axes:
        """metric can be delta_HR, percent_delta_HR, delta_HR_baseline"""

        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        for i, area in enumerate(_area_list):
            area_df = self.plot_data.filter(pl.col("area") == area)

            uniq_contrast = area_df["contrast"].drop_nulls().unique().sort().to_numpy()
            uniq_contrast = [c for c in uniq_contrast if c not in [100, 6.25, 0]]

            contrast_axis_shift = [-0.1, 0.1]
            for j, c in enumerate(uniq_contrast):

                # to plot single contrast or both
                if contrast == c:
                    contrast_ind = i
                elif contrast != -1 and contrast != c:
                    continue
                else:
                    contrast_ind = i + contrast_axis_shift[j]

                contrast_df = area_df.filter(pl.col("contrast") == c)

                uniq_sessions = area_df["session_no"].unique().to_list()
                area_resps = []
                colors = []
                for sesh in uniq_sessions:
                    sesh_df = contrast_df.filter(
                        (pl.col("session_no") == sesh) & (pl.col("stim_side") == "contra")
                    )

                    q = (
                        sesh_df.group_by(
                            ["stim_type", "contrast", "stim_side", "opto_pattern"]
                        )
                        .agg(
                            [
                                (pl.col("stim_pos").first()),
                                pl.count().alias("count"),
                                (pl.col("outcome") == 1).sum().alias("correct_count"),
                                (pl.col("outcome") == 0).sum().alias("miss_count"),
                                (pl.col("reaction_time").alias("response_times")),
                                (
                                    pl.when(pl.col("outcome") == 1)
                                    .then(pl.col("reaction_time"))
                                    .alias("response_times_correct")
                                ),
                            ]
                        )
                        .sort(["stim_type", "contrast", "stim_side", "opto_pattern"])
                    )
                    nonopto_resp = [
                        r for r in q[0, "response_times_correct"].to_numpy() if r < 1000
                    ]
                    opto_resp = [
                        r for r in q[1, "response_times_correct"].to_numpy() if r < 1000
                    ]

                    if not len(opto_resp):
                        opto_resp = [1000]

                    median_sesh_resp_diff = np.nanmedian(opto_resp) - np.nanmedian(
                        nonopto_resp
                    )
                    area_resps.append(median_sesh_resp_diff)

                    if kwargs.pop("color_animals", False):
                        colors.append(ANIMAL_COLORS[sesh_df[0, "animalid"]])
                    else:
                        _clr = "#bfbfbf" if c == 12.5 else "#424242"
                        colors.append(_clr)

                _x = self.add_jitter([contrast_ind] * len(area_resps))

                ax.scatter(_x, area_resps, s=100, color=colors, alpha=1, linewidths=0)

                # do violin
                if kwargs.get("violin_plot", False):
                    parts = ax.violinplot(
                        area_resps,
                        [contrast_ind],
                        showmedians=False,
                        showextrema=False,
                        widths=0.5,
                        side="low" if j == 0 else "high",
                    )

                    for pc in parts["bodies"]:
                        if j == 0:
                            pc.set_facecolor("#bfbfbf")
                        else:
                            pc.set_facecolor("#424242")
                        pc.set_alpha(0.6)

                quartile1, medians, quartile3 = np.percentile(area_resps, [25, 50, 75])
                ax.scatter(contrast_ind, medians, marker="_", color="k", s=100, zorder=3)
                ax.vlines(
                    contrast_ind,
                    quartile1,
                    quartile3,
                    color="k",
                    linestyle="-",
                    linewidth=1.5,
                )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel("delta_resp_time", fontsize=fontsize)

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list)))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.grid(alpha=0.4)

        return ax

    def plot_correlation_of_measures(
        self, ax: plt.Axes, hr_metric: str = "percent_delta_HR", **kwargs
    ) -> plt.Axes:
        """ """
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

    @staticmethod
    def make_dot_cloud(
        y: ArrayLike, center_pos: float = 0, nbins=None, width: float = 0.8
    ):
        """
        Returns x coordinates for the points in ``y``, so that plotting ``x`` and
        ``y`` results in a bee swarm plot.
        """
        y = np.asarray(y)
        if nbins is None:
            nbins = len(y) // 6
            print(nbins)

        # Get upper bounds of bins
        counts, bin_edges = np.histogram(y, bins=5)
        print(bin_edges)
        print(counts)

        # get the indices that correspond to points inside the bin edges
        ibs = []
        for ymin, ymax in zip(bin_edges[:-1], bin_edges[1:]):
            i = np.nonzero((y >= ymin) * (y < ymax))[0]
            ibs.append(i)

        x_coords = np.zeros(len(y))
        dx = width / (np.nanmax(counts) // 2)
        for i in ibs:
            _points = y[i]  # value of points that fall into the bin
            # if less then 2, leave untouched, will put it in the mid line
            if len(i) > 1:
                j = len(i) % 2
                i = i[np.argsort(_points)]
                # if even numbers of points, j will be 0, which will allocate the points equally to left and right
                # if odd, j will be 1, then, below lines will leave idx 0 at the midline and start from idx 1
                a = i[j::2]
                b = i[j + 1 :: 2]

                x_coords[a] = (0.5 + j / 3 + np.arange(len(a))) * dx
                x_coords[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

        return x_coords + center_pos

    def plot_raw_reaction_times(
        self,
        ax: plt.Axes = None,
        contrast: float = -1,
        reaction_of: str = "transformed",
        include_misses: bool = False,
        bin_width: int = 5,  # ms
        **kwargs,
    ) -> plt.Axes:
        """"""
        if ax is None:
            self.fig = plt.figure(figsize=kwargs.pop("figsize", (10, 10)))
            ax = self.fig.add_subplot(1, 1, 1)

        # order by area size
        if kwargs.pop("order_by_size", False):
            _sorted_dict = {
                k: v
                for k, v in sorted(
                    AREA_SIZE.items(), key=lambda item: item[1], reverse=True
                )
            }
            _area_list = list(_sorted_dict.keys())
        else:
            _area_list = self.area_list

        _offset_coeff = kwargs.pop("area_offset", 2)
        pattern_offset = [-_offset_coeff, _offset_coeff]
        _idx_coeff = (
            len(self.plot_data["opto_pattern"].drop_nulls().unique()) + _offset_coeff * 3
        )
        for i, area in enumerate(_area_list):
            idx = _idx_coeff * i
            area_df = self.plot_data.filter(
                (pl.col("area") == area)
                & (pl.col("contrast") == contrast)
                & (pl.col("stim_side") == "contra")
            )

            if include_misses == False:
                area_df = area_df.filter(pl.col("outcome") == 1)

            _both_rts = []
            for j, pattern in enumerate([-1, 0]):
                _rt = area_df.filter(pl.col("opto_pattern") == pattern)[
                    "reaction_time"
                ].to_list()
                _rt = [r for r in _rt if r > 150]

                _both_rts.append(_rt)
                _x = self.make_dot_cloud(
                    _rt, idx + pattern_offset[j], nbins=11, width=_offset_coeff / 2
                )

                ax.scatter(_x, _rt, s=160, c="k" if pattern == -1 else "#88CCEE")

                ax.hlines(
                    y=np.nanmedian(_rt),
                    xmin=idx + pattern_offset[j] - _offset_coeff,
                    xmax=idx + pattern_offset[j] + _offset_coeff,
                    linewidth=5,
                    color="w" if pattern == -1 else "b",
                )

            res = stats.kruskal(*_both_rts, nan_policy="omit")
            p = res.pvalue
            stars = ""
            if p < 0.0001:
                stars = "****"
            elif p < 0.001:
                stars = "***"
            elif 0.001 < p < 0.01:
                stars = "**"
            elif 0.01 < p < 0.05:
                stars = "*"
            else:
                continue
            ax.text(idx, 1000, stars, color="k")
            ax.text(
                idx,
                1100,
                round(p, 6),
                color="k",
            )

        ax.axhline(0, color="k", linestyle=":", linewidth=1)

        fontsize = 20
        ax.set_yscale("log")
        ax.set_xlabel("Area", fontsize=fontsize)
        ax.set_ylabel("Reaction Times", fontsize=fontsize)

        ax.tick_params(labelsize=fontsize, length=6)

        ax.set_xticks(np.arange(0, len(_area_list) * _idx_coeff, _idx_coeff))
        ax.set_xticklabels(_area_list)
        ax.tick_params(axis="x", rotation=45)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax

