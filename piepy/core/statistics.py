from __future__ import division
import numpy as np
from typing import Literal
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist, cdist
from scipy.stats import (
    bootstrap,
    sem,
    t,
    mannwhitneyu,
    wilcoxon,
    chi2,
    pearsonr,
    kstwobign
)


def mean_confidence_interval(data:ArrayLike, confidence:float=0.95) -> tuple[float,float]:
    """Calculates the mean and CI of it of given data

    Args:
        data (ArrayLike): 1D array of samples
        confidence (float, optional): Desired confidence interval. Defaults to 0.95.

    Returns:
        tuple[float,float]: mean and CI
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h


def bootstrap_confidence_interval(data:ArrayLike, statistic:callable, confidence:float=0.95,nboot:int=1000) -> tuple[float,float,float]:
    """Calculates the median and CI of the median of given data

    Args:
        data (ArrayLike): 1D array of samples
        confidence (float, optional): desired confidence interval. Defaults to 95%.
        nboot (int, optional): Amount of resamples. Defaults to 1000.

    Returns:
        tuple[float,float,float]: median, +CI and -CI 
    """
    a = 1.0 * np.array(data)
    a = a[~np.isnan(a)]
    _med = statistic(a)
    res = bootstrap((a,),statistic,n_resamples=nboot,confidence_level=0.95,method="bca")
    
    m = np.mean(res.bootstrap_distribution)
    
    ci_plus = res.confidence_interval.high - m
    ci_neg = m - res.confidence_interval.low

    return _med, ci_plus, ci_neg


def nonparametric_pvalues(x1:ArrayLike, x2:ArrayLike, method: Literal["mannu","wilcoxon"]="mannu") -> float:
    """Returns the significance value of two distributions 
    
    Args:
        x1 (ArrayLike): First set of samples
        x2 (ArrayLike): Second set of samples
        method (Literal["mannu","wilcoxon"], optional): Non-parametric test method. Defaults to "mannu".

    Raises:
        ValueError: Invalid statistical test method

    Returns:
        float: p-value
    """
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


def ks2s_2d(
    data1:ArrayLike, 
    data2:ArrayLike, 
    nboot:int|None=None
    ) -> tuple[float,float]:
    """Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Adapted from: https://github.com/syrte/ndtest
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. 
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation
    is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate,
    but it certainly implies that the two samples are not significantly different. 

    Args:
        data1 (ArrayLike): shape (n1,2) Data of sample one
        data2 (ArrayLike): shape (n2,2) Data of sample two (n1 and n2 can be different)
        nboot (int | None, optional): Number of bootstrap resample to estimate the p-value. A large number is expected.
        If None, an approximate analytic estimate will be used. Defaults to None.

    Returns:
        tuple[float,float]: Two-tailed p-value, KS statistic
    """
    def quadct(x, y, xx, yy):
        n = len(xx)
        ix1, ix2 = xx <= x, yy <= y
        a = np.sum(ix1 & ix2) / n
        b = np.sum(ix1 & ~ix2) / n
        c = np.sum(~ix1 & ix2) / n
        d = 1 - a - b - c
        return a, b, c, d
    
    def maxdist(x1, y1, x2, y2):
        n1 = len(x1)
        D1 = np.empty((n1, 4))
        for i in range(n1):
            a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
            a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
            D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

        # re-assign the point to maximize difference,
        # the discrepancy is significant for N < ~50
        D1[:, 0] -= 1 / n1

        dmin, dmax = -D1.min(), D1.max() + 1 / n1
        return max(dmin, dmax)
    
    def avgmaxdist(x1,y1,x2,y2):
        D1 = maxdist(x1,y1,x2,y2)
        D2 = maxdist(x2,y2,x1,y1)
        return (D1 + D2) / 2

    n1, n2 = len(data1), len(data2)
    
    x1, x2 = data1[:,0], data2[:,0]
    y1, y2 = data1[:,1], data2[:,1]
    D = avgmaxdist(x1,y1,x2,y2)
    
    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = np.random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot

    return p, D


def energy_stat_2d(
    data1:ArrayLike,
    data2:ArrayLike,
    nboot:int=1000,
    replace:bool=False,
    method:Literal["log","gaussian","linear"]='log'
    ) -> tuple[float, float, float]:
    """Energy distance statistics test.
    Adapted from: https://github.com/syrte/ndtest

    Args:
        data1 (ArrayLike): shape (n1,2) Data of sample one
        data2 (ArrayLike): shape (n2,2) Data of sample two (n1 and n2 can be different)
        nboot (int, optional): Number of bootstrap resample to estimate the p-value. A large number is expected. Defaults to 1000.
        replace (bool, optional): Sample with replacement. Defaults to False.
        method (Literal["log","gaussian","linear"], optional): . Defaults to 'log'.

    Returns:
        tuple[float, float, float]: p-value, energy,
    """
    
    def energy(x, y, method='log'):
        dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
        n, m = len(x), len(y)
        if method == 'log':
            dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
        elif method == 'gaussian':
            raise NotImplementedError
        elif method == 'linear':
            pass
        else:
            raise ValueError
        z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
        # z = ((n*m)/(n+m)) * z # ref. SR
        return z
    
    n, N = len(data1), len(data1) + len(data2)
    stack = np.vstack((data1, data2))
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: np.random.randint(x, size=x)  # noqa: E731
    else:
        rand = np.random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    # if fitting:
    #     param = genextreme.fit(en_boot)
    #     p = genextreme.sf(en, *param)
    #     return p, en, param
    # else:
    p = (en_boot >= en).sum() / nboot
    return p, en, en_boot


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

