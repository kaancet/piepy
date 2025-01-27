import polars as pl
import numpy as np
from typing import Literal
from scipy.stats import (
    chi2,
    mannwhitneyu,
    wilcoxon,
    fisher_exact,
    barnard_exact,
    boschloo_exact,
)

from ..core.data_functions import make_subsets


class GroupedAggregator:
    def __init__(self) -> None:
        pass

    def set_data(self, data: pl.DataFrame, outcomes:list, **kwargs) -> None:
        """ Sets the data to be grouped/aggregated and statsed on
        
        Args:
            data: DataFrame of the run/session
            outcomes: list of outcomes to aggregate under
        """
        self.data = data
        self.grouped_data = self.group_data(outcomes,
            kwargs.get("group_by", None), kwargs.get("do_sort", True)
        )
        # self.calculate_hit_rates(kwargs.get("p_method", "barnard"))

    def group_data(
        self, outcomes:list, group_by: list[str] = None, do_sort: bool = True
    ) -> pl.DataFrame:
        """Groups the data by given group_by column names
        
        Args:
            outcomes: list of outcomes to aggregate under
            group_by: List oo=f column names to group the data by
            do_sort: falg to indicate whether to sort the data in the order of group_by
        """

        for o in outcomes:
            if o not in self.data["outcome"]:
                raise ValueError(f"{o} is not a valid outcome type that is present in the outcome column of the data")

        if group_by is None:
            group_by = ["stim_type", "contrast", "stim_side", "opto_pattern"]

        for c_name in group_by:
            if c_name not in self.data.columns:
                raise ValueError(f"{c_name} not in data columns!!")

        q = self.data.group_by(group_by).agg(
            [
                (pl.col("stim_pos").first()),
                pl.count().alias("count"),
            ]+ 
            [
                (pl.col("outcome")==o).sum().alias(f"{o}_count") for o in outcomes
            ]+
            [
                (pl.col("response_time").alias("response_times")),
                (pl.col("rig_response_time").alias("rig_response_times")),
                (pl.col("reaction_time").alias("reaction_times")),
            ]+
            [
                (
                    pl.col("response_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_response_times")
                ) for o in outcomes
            ]+
            [
                (
                    pl.col("response_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"{o}_median_response_time")
                ) for o in outcomes
            ] +
            [
                (
                    pl.col("rig_response_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_rig_response_times")
                ) for o in outcomes
            ]+
            [
                (
                    pl.col("rig_response_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"{o}_median_rig_response_time")
                ) for o in outcomes
            ] + 
            [
                (
                    pl.col("reaction_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_reaction_times")
                )
            ] +
            [
                (
                    pl.col("reaction_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"{o}_median_reaction_time")
                ) for o in outcomes
            ]+
            [
                (pl.col("wheel_t")),
                (pl.col("wheel_pos")),
                (pl.col("signed_contrast").first()),
                (pl.col("opto").first()),
                (pl.col("stimkey").first()),
                (pl.col("stim_label").first()),
            ]
        )

        if do_sort:
            q = q.sort(group_by)
        return q

    def calculate_hit_rates(self, 
                            positive:str, 
                            negative:str,
                            p_method: Literal["barnard", "boschloo", "fischer"] = "barnard") -> None:
        """Sets the hit rates for each condition based on binomial distribution of hit count,
        Needs data to be grouped first
        
        Args:
            positive: name to positive outcome (hit, correct)
            negative: name of the negative outcome (miss, incorrect)
            p_method: method to calculate the p-value
        """
        # hit rates
        self.grouped_data = self.grouped_data.with_columns(
            (pl.col(f"{positive}_count") / pl.col("count")).alias("hit_rate")
        )
        self.grouped_data = self.grouped_data.with_columns(
            (
                1.96
                * np.sqrt(
                    (pl.col("hit_rate") * (1.0 - pl.col("hit_rate"))) / pl.col("count")
                )
            ).alias("confs")
        )

        # p-values
        p_vals = []
        non_early = self.grouped_data.filter(pl.col("stim_type").is_not_null())
        for filt_tup in make_subsets(non_early, ["stim_type", "contrast", "stim_side"]):
            _df = filt_tup[-1]
            p = None
            if 0 not in _df["opto_pattern"]:
                # print("CAN'T DO P-VALUE ANALYSIS, MISSING OPTO COMPONENTS!! RETURNING AN EMPTY DATAFRAME")
                pass

            if len(_df):
                table = _df[:, [f"{positive}_count", f"{negative}_count"]].to_numpy()
                if table.shape == (2, 2) and not np.any(np.isnan(table)):
                    # all elements are filled
                    if p_method == "barnard":
                        res = barnard_exact(table, alternative="two-sided")
                    elif p_method == "boschloo":
                        res = boschloo_exact(table, alternative="two-sided")
                    elif p_method == "fischer":
                        res = fisher_exact(table, alternative="two-sided")
                    p = res.pvalue
                    p_vals.extend([None, p])
                else:
                    p_vals.extend([p] * len(_df))
            else:
                p_vals.extend([p] * len(_df))

        len_diff = len(self.grouped_data) - len(p_vals)
        if len_diff < 0:
            raise
        p_vals = [None] * len_diff + p_vals
        # p values are ordered because make_subsets sorts the dataframe and then runs through it
        self.grouped_data = self.grouped_data.with_columns(
            pl.Series("p_hit_rate", p_vals)
        )
        
    @staticmethod
    def get_pvalues_nonparametric(x1, x2, method: Literal["mannu","wilcoxon"] = "mannu") -> dict:
        """Returns the significance value of two distributions"""
        if method not in ["mannu", "wilcoxon"]:
            raise ValueError(
                f"{method} not a valid statistical test yet, try mannu or wilcoxon"
            )
        if method == "mannu":
            _, p = mannwhitneyu(x1, x2)
        elif method == "wilcoxon":
            res = wilcoxon(x1, x2)
            p = res.p
        return p


def mantel_haenzsel(data, stats="Q"):
    """Garcia-Perez, M.A., & Nuñez-Anton, V. (2018). Nonparametric tests for
    equality of psychometric functions. Behavior Research Methods,
    https://doi.org/10.3758/s13428-017-0989-0"""
    # error handling
    if not isinstance(data, dict):
        raise TypeError("data argument needs a dict, got a {0}".format(type(data)))

    num_tables = len(data)
    if isinstance(stats, str):
        if stats != "Q" and stats != "G":
            raise ValueError("stats string either takes Q or G not {0}".format(stats))
    else:
        if num_tables < 2:
            raise ValueError(
                "The split Mantel-Haenszel test cannot be used if there is only one table"
            )
        elif len(stats) != 2:
            raise ValueError(
                "stats vector should have 2 elements, got {0}".format(len(stats))
            )
        elif any(elem <= 0 for elem in stats):
            raise ValueError(
                "Elements in stats vector should be positive, got {0}".format(stats)
            )
        elif sum(stats) != num_tables:
            raise ValueError(
                "stats vector does not add up to ({0}) length of data({1})".format(
                    stats, num_tables
                )
            )

    table_ids = list(data.keys())
    cr = np.array([np.nan] * num_tables)
    cc = np.array([np.nan] * num_tables)

    for it, key in enumerate(data.keys()):
        # fill in row and column sizes for each data table
        cr[it] = data[key].shape[0]
        cc[it] = data[key].shape[1]
        # check dimensions of tables
        if data[key].ndim > 2 or cr[it] < 2 or cc[it] < 2:
            raise ValueError("Invalid table at {0}, not a two-way table".format(key))
        # check if any table element is 0 or fractional
        elif (data[key] < 0).any(axis=None) or not (data[key] % 1 == 0).all(axis=None):
            raise ValueError(
                "Invalid table at {0}, contains negative or fractional values".format(key)
            )

    if np.sum(np.add(cr, -cr[0])) != 0:
        raise ValueError("Invalid Data, tables do not have the same number of rows")

    if np.sum(np.add(cc, -cc[0])) != 0:
        raise ValueError("Invalid Data, tables do not have the same number of columns")

    cc = cc.astype(int)
    cr = cr.astype(int)

    # compute selected statistic
    if isinstance(stats, str):
        kase = 1 if stats == "G" else 2
    else:
        kase = 3

    # generalized Berry-Mielke test
    if kase == 1:
        pass
        # N_k = np.array([np.nan] * num_tables)
        # Tmean = np.array([np.nan] * num_tables)
        # Tvar  = np.array([np.nan] * num_tables)
        # Tskew = np.array([np.nan] * num_tables)
        # Tstat = np.array([np.nan] * num_tables)
        # Zstat = np.array([np.nan] * num_tables)
        # Gstat = np.array([np.nan] * num_tables)
        # c_k   = np.array([np.nan] * num_tables)
        # Ncols = np.ones(num_tables,dtype=bool)
        # Nrows = np.ones(num_tables,dtype=bool)
        # Ntbls = np.ones(num_tables,dtype=bool)
        # sngl_row = np.zeros(num_tables,dtype=bool)
        # sngl_col = np.zeros(num_tables,dtype=bool)
        # sngl_perm = np.zeros(num_tables,dtype=bool)
        # data_used = {}

        # warn1 = 'All tables were used'
        # warn2 = 'All usable tables were used with their numbers of rows'
        # warn3 = 'All usable tables were used with their numbers of columns'
        # warn4 = 'For all usable tables, gamma_T >= 0.5'

        # for it,key in enumerate(data.keys()):
        #     in_data = data[it]
        #     row = np.sum(in_data,1).to_numpy()
        #     col = np.sum(in_data,0).to_numpy()

        #     N = np.sum(row);
        #     N_k[it] = N
        #     # check for actual number of rows and columns
        #     table = in_data.loc[row>0,col>0]

        #     row = row[row>0]
        #     col = col[col>0]
        #     I = len(row)
        #     J = len(col)

        #     if I==1:
        #         sngl_row[it] = True
        #     if J==1:
        #         sngl_co[it] = True

        #     if table.shape[0] > 1 and table.shape[0] != in_data.shape[0]:
        #         Nrows[it] = False

        #     if table.shape[1] > 1 and table.shape[1] != in_data.shape[1]:
        #         Ncols[it] = False

        #     # check for Ix2 table with equal row marginal frequencies and a column marginal frequency of 1
        #     if J == 1 or (J == 2 and all(row == row[0]) and any(col == 1)):
        #         Ntbls[it] = False;
        #         sngl_perm[it] = True

        #     # check for 2xJ table with equal column marginal frequencies and a row marginal frequency of 1
        #     if I == 1 or (I == 2 and all(col == col[0]) and any(row == 1)):
        #         Ntbls[it] = False
        #         sngl_perm[it] = True

        #     if Ntbls[it]:
        #         data_used[key] = table
        #         # compute moments
        #         N_ = np.empty((6,2))
        #         N_[:] = np.nan
        #         R_m = np.empty((I,6))
        #         R_m[:] = np.nan
        #         C_m = np.empty((J,6))
        #         C_m[:] = np.nan
        #         R = np.empty((6,6))
        #         R[:] = np.nan
        #         C = np.empty((6,6))
        #         C[:] = np.nan

        #         for m in range(4):
        #             N_[m,0] = np.prod(N-3:N-4+m)
        #         for m in range(6):
        #             N_[m,1] = np.prod(N-5:N-6+m)

        #         R_m[:,0] = row
        #         for m in range(1,6):
        #             R_m[:,m] = R_m[:,m-1] * (row - m +1)

        #         for m in range(4):
        #             R_m[m,1] = np.sum(R_m[:,m] / (row ** 2))

        #         R[2,2] = I * (I-1)
        #         R[3,2] = (I-1) * (N-I)
        #         R[4,2] = (N-I)**2 + 2 * N -I np.sum(row**2)

        #         for m in range(6):
        #             R[m,3] = np.sum(R_m[:,m] / (row**3))

        #         for m in range(2,6):
        #             R[m,4] = np.sum(R_m[:,m-2] * (N-row-I+1)/(row**2))

        #         for m

        # for m=3:6, R(m,4) = sum(R_m(:,m-2).*(N-row-I+1)./(row.^2)); end
        # for m=2:5, R(m,5) = (I-1)*R(m-1,1); end
        # R(3,6) = I*(I-1)*(I-2);
        # R(4,6) = (I-1)*(I-2)*(N-I);
        # R(5,6) = (I-2)*R(4,2);

    # generalized Mantel-Haenzsel test
    elif kase == 2:
        description = "Generalized Mantel-Haenszel test in {0} populations with {1} response categories".format(
            cr[0], cc[0]
        )

        R = {k: None for k in table_ids}
        C = {k: None for k in table_ids}
        N = np.array([np.nan] * num_tables)
        O = data.copy()
        data_used = data.copy()
        E = {k: None for k in table_ids}
        V = {k: None for k in table_ids}
        row = np.zeros((cr[0], 1), dtype=int)
        col = np.zeros((1, cc[0]), dtype=int)

        # check for empty rows or columns accross tables
        for it, key in enumerate(data.keys()):
            R[key] = np.sum(data[key], axis=1).reshape(cr[0], -1)
            C[key] = np.sum(data[key], axis=0).reshape(-1, cc[0])
            N[it] = np.sum(C[key])
            E[key] = np.matmul(R[key], C[key]) / N[it]
            row += R[key]
            col += C[key]

        I = np.sum(row > 0)
        J = np.sum(col > 0)

        warn_msg1 = ""
        warn_msg2 = ""
        qgmh_stat = None
        df = None
        p_value = None

        if I == 1 or J == 1:
            warn_msg1 = "Data cannot be used: all but one column or one row are empty"
        else:
            if I < cr[0] or J < cc[0]:
                warn_msg1 = "Tables used with {0} rows and {1} columns".format(I, J)
            else:
                warn_msg1 = "Tables used with all their rows and columns"

            if num_tables == 1:
                warn_msg2 = "Q_GMH is only an adjusted Pearson" "s statistic when K = 1"

            sum1 = np.zeros((1, (I - 1) * (J - 1)))
            sum2 = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

            for it, key in enumerate(table_ids):
                R[key] = R[key][row > 0]
                R[key] = R[key][0 : I - 1]
                R[key] = R[key].reshape(len(R[key]), -1)
                C[key] = C[key][col > 0]
                C[key] = C[key][0 : J - 1]
                C[key] = C[key].reshape(-1, len(C[key]))

                E[key] = E[key][row[:, 0] > 0, :]  # row filter
                E[key] = E[key][:, col[0, :] > 0]  # column
                E[key] = np.reshape(
                    E[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F"
                )

                O[key] = O[key][row[:, 0] > 0, :]  # row filter
                O[key] = O[key][:, col[0, :] > 0]  # column
                data_used[key] = O[key]

                O[key] = np.reshape(
                    O[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F"
                )

                sum1 += np.subtract(O[key], E[key])
                if N[it] > 1:
                    V[key] = np.kron(
                        N[it] * np.diag(C[key][0])
                        - np.matmul(np.transpose(C[key]), C[key]),
                        N[it] * np.diag(R[key]) - np.matmul(R[key], np.transpose(R[key])),
                    ) / (N[it] * N[it] * (N[it] - 1))
                else:
                    V[key] = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                sum2 = np.add(sum2, V[key])

            if np.linalg.matrix_rank(sum2) == sum2.shape[0]:
                temp = np.linalg.lstsq(sum2.T, sum1.T)[0]
                qgmh_stat = np.matmul(temp.T, sum1.T)
                df = (I - 1) * (J - 1)
                p_value = chi2.sf(qgmh_stat, df)

        output = {
            "NumTables": num_tables,
            "SampleSizes": N,
            "Warning1": warn_msg1,
            "warning2": warn_msg2,
            "Q_GMH": qgmh_stat,
            "deg_free": df,
            "p_value": p_value,
        }

    # split Mantel-Haenszel test
    elif kase == 3:
        description = "Split Mantel-Haenszel test in {0} populations with {1} response categories".format(
            cr[0], cc[0]
        )

        R = {k: None for k in table_ids}
        C = {k: None for k in table_ids}
        N = np.array([np.nan] * num_tables)
        O = data.copy()
        data_used = data.copy()
        E = {k: None for k in table_ids}
        V = {k: None for k in table_ids}
        qgmh_stat = np.array([np.nan] * 2)
        df = np.array([np.nan] * 2)
        warn_msg1 = {0: None, 1: None}
        cual = {0: "first split", 1: "second split"}
        first = [0, stats[0]]
        last = [stats[0], num_tables]
        for split in range(2):
            row = np.zeros((cr[0], 1), dtype=int)
            col = np.zeros((1, cc[0]), dtype=int)
            for it in range(first[split], last[split]):
                key = list(data.keys())[it]
                R[key] = np.sum(data[key], axis=1).reshape(cr[0], -1)
                C[key] = np.sum(data[key], axis=0).reshape(-1, cc[0])
                N[it] = np.sum(C[key])
                E[key] = np.matmul(R[key], C[key]) / N[it]
                row += R[key]
                col += C[key]

            I = np.sum(row > 0)
            J = np.sum(col > 0)

            if I == 1 or J == 1:
                warn_msg1[split] = (
                    "Tables in {0} cannot be used: all but one column or one row are empty".format(
                        cual[split]
                    )
                )
            else:
                if I < cr[0] or J < cc[0]:
                    warn_msg1[split] = (
                        "Tables in {0} used with {1} rows and {2} columns".format(
                            cual[split], I, J
                        )
                    )
                else:
                    warn_msg1[split] = (
                        "Tables in {0} used with all their rows and columns".format(
                            cual[split]
                        )
                    )
                warn_msg2 = ""

                if num_tables == 1:
                    warn_msg2 = (
                        "S-Q_GMH is only the sum of adjusted Pearson"
                        "s statistics when K = 2"
                    )

                sum1 = np.zeros((1, (I - 1) * (J - 1)))
                sum2 = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                for it in range(first[split], last[split]):
                    key = list(data.keys())[it]
                    R[key] = R[key][row > 0]
                    R[key] = R[key][0 : I - 1]
                    R[key] = R[key].reshape(len(R[key]), -1)
                    C[key] = C[key][col > 0]
                    C[key] = C[key][0 : J - 1]
                    C[key] = C[key].reshape(-1, len(C[key]))

                    E[key] = E[key][row[:, 0] > 0, :]  # row filter
                    E[key] = E[key][:, col[0, :] > 0]  # column
                    E[key] = np.reshape(
                        E[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F"
                    )

                    O[key] = O[key][row[:, 0] > 0, :]  # row filter
                    O[key] = O[key][:, col[0, :] > 0]  # column
                    data_used[key] = O[key]

                    O[key] = np.reshape(
                        O[key][0 : I - 1, 0 : J - 1], (1, (I - 1) * (J - 1)), order="F"
                    )

                    sum1 += np.subtract(O[key], E[key])
                    if N[it] > 1:
                        V[key] = np.kron(
                            N[it] * np.diag(C[key][0])
                            - np.matmul(np.transpose(C[key]), C[key]),
                            N[it] * np.diag(R[key])
                            - np.matmul(R[key], np.transpose(R[key])),
                        ) / (N[it] * N[it] * (N[it] - 1))
                    else:
                        V[key] = np.zeros(((I - 1) * (J - 1), (I - 1) * (J - 1)))

                    sum2 = np.add(sum2, V[key])

                if np.linalg.matrix_rank(sum2) == sum2.shape[0]:
                    temp = np.linalg.lstsq(sum2.T, sum1.T)[0]
                    qgmh_stat[split] = np.matmul(temp.T, sum1.T)
                    df[split] = (I - 1) * (J - 1)

        s_qgmh_stat = np.nansum(qgmh_stat)
        s_df = np.nansum(df)
        p_value = None
        if not np.isnan(s_qgmh_stat).all():
            p_value = chi2.sf(s_qgmh_stat, s_df)

        output = {
            "description": description,
            "NumTables": num_tables,
            "In_Split1": [*range(0, stats[0])],
            "In_Split2": [*range(stats[0], num_tables)],
            "SampleSizes": N,
            "Warning1": warn_msg1,
            "warning2": warn_msg2,
            "Components": qgmh_stat,
            "S-Q_GMH": s_qgmh_stat,
            "deg_free": s_df,
            "p_value": p_value,
        }

    return output, data_used
