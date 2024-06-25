import time
from tqdm import tqdm
import pandas as pd

from scipy.optimize import curve_fit
from ..wheelUtils import *
from ..fit_funcs import *
from ..model.transformer import GLMHMMTransfromer
from ..model.glmModel import Glm
from ..utils import display, save_dict_json
from ..statistics import *
from .wheelSession import *


class WheelCurve:
    """Simple curve fitting on the data"""

    __slots__ = [
        "data",
        "fit_model",
        "fit_side",
        "contrast",
        "signed_contrast",
        "percentage",
        "confidence",
        "counts",
        "fit_pars",
        "fitted_curve",
        "bias",
    ]

    def __init__(
        self,
        data: pd.DataFrame = None,
        fit_model: str = "erf_psycho2",
        fit_side: str = "right",
        init_dict: dict = None,
    ):
        if init_dict is None:
            self.data = data
            self.fit_model = fit_model
            if fit_side not in ["left", "nogo", "right"]:
                raise ValueError(f"Psychometric fitting wrt {fit_side} is not possible")
            self.fit_side = fit_side
            self.percentage = None
            self.confidence = None
        else:
            self.init_from_dict(init_dict)

    def init_from_dict(self, init_dict: dict) -> None:
        for k, v in init_dict.items():
            setattr(self, k, v)

    def get_contrast(self):
        """Gets the signed contrast list and the counts for each contrast value on each side"""
        self.contrast = np.sort(self.data["contrast"].unique())
        self.data.loc[:, "signed_contrast"] = self.data.loc[
            :, ["contrast", "stim_side"]
        ].apply(lambda row: row[0] * np.sign(row[1]) if row[0] != 0 else row[0], axis=1)
        self.signed_contrast, self.counts = np.unique(
            self.data["signed_contrast"], return_counts=True
        )

    def get_percentages(self):
        """Calculates the percentages and their confidence"""
        if self.percentage is None:
            # if a curve has already been fitted
            self.get_contrast()

            if self.fit_side == "left":
                answer_check = -1
            elif self.fit_side == "nogo":
                answer_check = 0
            else:
                answer_check = 1

            for contrast in tqdm(self.contrast, desc="calculating percentages"):
                contrast_data = self.data[self.data["contrast"] == contrast]

                # right side
                right_contrast_data = contrast_data[contrast_data["stim_side"] > 0]
                cnt_on_right = len(right_contrast_data)
                if cnt_on_right > 0:
                    perc_on_right = (
                        len(
                            right_contrast_data[
                                right_contrast_data["answer"] == answer_check
                            ]
                        )
                        / cnt_on_right
                    )
                    conf_on_right = 1.96 * np.sqrt(
                        (perc_on_right * (1 - perc_on_right)) / cnt_on_right
                    )  # 95% binomial
                else:
                    perc_on_right = None
                    conf_on_right = None
                # left side
                left_contrast_data = contrast_data[contrast_data["stim_side"] < 0]
                cnt_on_left = len(left_contrast_data)
                if cnt_on_left > 0:
                    perc_on_left = (
                        len(
                            left_contrast_data[
                                left_contrast_data["answer"] == -1 * answer_check
                            ]
                        )
                        / cnt_on_left
                    )
                    conf_on_left = 1.96 * np.sqrt(
                        (perc_on_left * (1 - perc_on_left)) / cnt_on_left
                    )  # 95% binomial
                else:
                    perc_on_left = None
                    conf_on_left = None

                if contrast == 0:
                    cnt_on_zero = cnt_on_left + cnt_on_right
                    perc_on_zero = (
                        len(
                            left_contrast_data[
                                left_contrast_data["answer"] == -1 * answer_check
                            ]
                        )
                        + len(
                            right_contrast_data[
                                right_contrast_data["answer"] == answer_check
                            ]
                        )
                    ) / cnt_on_zero
                    conf_on_zero = 1.96 * np.sqrt(
                        (perc_on_zero * (1 - perc_on_zero)) / cnt_on_zero
                    )  # 95% binomial
                    self.percentage = [perc_on_zero]
                    self.confidence = [conf_on_zero]
                else:
                    if self.percentage is None:
                        self.percentage = []
                        self.confidence = []

                    if perc_on_left is not None:
                        self.percentage.insert(0, perc_on_left)
                        self.confidence.insert(0, conf_on_left)

                    if perc_on_right is not None:
                        self.percentage.append(perc_on_right)
                        self.confidence.append(conf_on_right)

            self.percentage = np.asarray(self.percentage)
            self.confidence = np.asarray(self.confidence)

    def fit_curve(self, **kwargs):
        """Fits the curve"""
        self.get_percentages()
        x_axis = np.linspace(
            np.min(self.signed_contrast),
            np.max(self.signed_contrast),
            kwargs.get("resolution", 100),
        ).reshape(-1, 1)

        if self.fit_model == "naive":
            try:
                popt, pcov = naive_fit(self.signed_contrast, self.percentage, **kwargs)

                self.fit_pars = {"pars": popt, "pcov": pcov}

                y_axis = np.asarray(sigmoid(x_axis, *self.fit_pars["pars"])).reshape(
                    -1, 1
                )

            except:
                display("Failed to fit to data")
                self.fit_pars = {"pars": 0, "popt": 0}
                x_axis = self.signed_contrast.reshape(-1, 1)
                y_axis = self.percentage.reshape(-1, 1)

        elif self.fit_model == "erf_psycho2":
            data = np.vstack((self.signed_contrast, self.counts, self.percentage))
            # try:
            fit_pars, fit_L = mle_fit(
                data, self.fit_model, side=self.fit_side, nfits=kwargs.get("nfits", 10)
            )

            self.fit_pars = {"pars": fit_pars, "likelihood": fit_L}

            y_axis = np.asarray(erf_psycho2(self.fit_pars["pars"], x_axis)).reshape(-1, 1)
            # except:
            # display('Failed to fit to data')
            # x_axis = self.signed_contrast.reshape(-1,1)
            # y_axis = self.percentage.reshape(-1,1)

        self.fitted_curve = np.hstack((x_axis, y_axis))


class WheelAnalysis(object):
    def __init__(self, data=None):
        self.analysis = {}
        self.pairs = {}
        if data is not None:
            self.set_data(data)

    def set_data(self, data, keep_overall=False):
        if keep_overall:
            self.data = {k: v for k, v in data.items()}
        else:
            self.data = {k: v for k, v in data.items() if k != "overall"}

        self.pre_analysis()

    def pre_analysis(self):
        self.analysis["summary"] = {}
        for i, key in enumerate(self.data.keys()):
            data_curr = self.data[key]
            summary_dict = {}
            summary_dict["contrast_list"] = np.unique(data_curr["contrast"]).tolist()
            # replace left side contrasts with negative value
            data_curr.loc[:, "sided_contrast"] = data_curr.loc[
                :, ["contrast", "stim_side"]
            ].apply(
                lambda row: row[0] * np.sign(row[1]) if row[0] != 0 else row[0], axis=1
            )
            summary_dict["sided_contrast_list"] = np.unique(
                data_curr["sided_contrast"]
            ).tolist()

            # get general counts and create a data table using those
            summary_dict["contrast_count"] = []
            summary_dict["left_count"] = []
            summary_dict["right_count"] = []
            for j, contrast in enumerate(summary_dict["sided_contrast_list"]):
                contrast_data = data_curr[data_curr["sided_contrast"] == contrast]
                # total count for contrast
                summary_dict["contrast_count"].append(len(contrast_data))
                # left count
                summary_dict["left_count"].append(
                    len(contrast_data[contrast_data["stim_side"] < 0])
                )
                # right count
                summary_dict["right_count"].append(
                    len(contrast_data[contrast_data["stim_side"] > 0])
                )

            self.analysis["summary"][key] = summary_dict

    def run_analysis(self, analysis_type=None, *args, **kwargs):
        """Loops through the conditions(in this case contrasts) and runs various analysis for individual datasets
        Also, creates a data table for each analysis
        """

        if len(self.data.keys()) < 2:
            display(
                "Only one type of stimulus present, skipping statistical comparison analysis"
            )
            return 0

        if analysis_type == "mantel_haenzsel":
            analyz_arg = kwargs.get("{0}_rgs".format(analysis_type), "Q")
            analyz = analysis_type + "_" + analyz_arg

        self.analysis[analyz] = {}
        display(" {0} ANALYSIS".format(analyz.upper()))

        self.analysis[analyz]["data_types"] = list(self.data.keys())

        # create unique, non-repeating pairs of stimulus type
        self.pairs = {}
        for i, _ in enumerate(self.analysis[analyz]["data_types"]):
            for j in range(1, len(self.analysis[analyz]["data_types"])):
                try:
                    key = "pair" + str(i) + str(j)
                    self.pairs[key] = [
                        self.analysis[analyz]["data_types"][i],
                        self.analysis[analyz]["data_types"][i + j],
                    ]
                except:
                    pass

        analysis_start = time.time()
        for i, p_key in enumerate(self.pairs.keys()):
            self.analysis[analyz][p_key] = {}
            self.analysis[analyz][p_key]["data_table"] = {}
            if "mantel_haenzsel" in analyz:
                display("Prepearing paired data table for {0}".format(p_key))
                pair = self.pairs[p_key]
                for j, typ in enumerate(pair):
                    for k, contrast in enumerate(
                        self.analysis["summary"][typ]["sided_contrast_list"]
                    ):
                        contrast_data = self.data[typ][
                            self.data[typ]["sided_contrast"] == contrast
                        ]
                        # correct answer count
                        count_correct = len(contrast_data[contrast_data["answer"] == 1])
                        # incorrect answer count
                        count_incorrect = len(
                            contrast_data[contrast_data["answer"] == -1]
                        )
                        # no answer count
                        count_noanswer = len(contrast_data[contrast_data["answer"] == 0])

                        table_row = np.array(
                            [count_correct, count_incorrect, count_noanswer]
                        ).reshape(1, 3)

                        # first pass, initiating the table
                        if j == 0:
                            self.analysis[analyz][p_key]["data_table"][
                                contrast
                            ] = table_row
                        else:
                            self.analysis[analyz][p_key]["data_table"][contrast] = (
                                np.vstack(
                                    (
                                        self.analysis[analyz][p_key]["data_table"][
                                            contrast
                                        ],
                                        table_row,
                                    )
                                )
                            )

                display("Running analysis {0} ...".format(p_key))
                self.analysis[analyz][p_key]["results"], _ = mantel_haenzsel(
                    data=self.analysis[analyz][p_key]["data_table"], stats=analyz_arg
                )

            elif "wilcoxon" in analyz:
                # get correct performances in each contrast for different stimuli groups
                display("Prepearing data tables")
                for i, typ in enumerate(self.analysis[analyz]["data_types"]):

                    percent_correct = []
                    for j, contrast in enumerate(
                        self.analysis["summary"][typ]["sided_contrast_list"]
                    ):
                        contrast_data = self.data[typ][
                            self.data[typ]["contrast"] == contrast
                        ]

                        # correct answer percentage
                        percent_correct.append(
                            len(contrast_data[contrast_data["answer"] == 1])
                            / len(contrast_data)
                        )

                    self.analysis[analyz]["data_table"][typ] = percent_correct

                # make a combination of key pairs in data_table and run wilcoxon on each pair
                stats.wilcoxon(a, b)

        # display('FINISHED {0} ANALYSIS, t={1}'.format(analyz.upper(),time.time()-analysis_start))


class WheelModel:
    __slots__ = ["model", "model_params", "n_init", "_model_data", "input_matrix", "y"]

    def __init__(self, model: str, n_init: int = 10, **kwargs) -> None:
        self.model = model
        self.n_init = n_init
        self.input_matrix = None
        self.y = None

    def __repr__(self) -> str:
        return f"{self.model.upper()} model for Wheel Task"

    @property
    def model_data(self):
        return self._model_data

    @model_data.setter
    def model_data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Data needs to be a DataFrame [trials x features], got {type(data)} instead"
            )
        self._model_data = data

    def transform_data(self, data_in: pd.DataFrame = None) -> None:

        if self.model in ["glm", "glmhmm"]:
            t = GLMHMMTransfromer()
            t.input_data = data_in
            t.transform_data()
            input_matrix, y, rewarded = t.get_session_unnormalized_data()

            normalized_input = np.copy(input_matrix)
            normalized_input[:, 0] = preprocessing.scale(normalized_input[:, 0])

            y = y.astype("int")
            go_idx = [i for i, row in enumerate(y) if row != -1]
            y = y[go_idx]
            normalized_input = normalized_input[go_idx]

            self.input_matrix = normalized_input
            self.y = y

    def fit_model(self, seed_init: int = 65):
        """Fits the selected model seed_init times and returns
        the parameters of the most likely one
        """
        if self.input_matrix is None and self.y is None:
            raise ValueError(
                "Input matrix and the output(y) has to be set before fitting the model"
            )

        log_likelihood = []
        models = []

        np.random.seed(seed_init)
        C = len(np.unique(y))
        M = self.input_matrix.shape[1]
        for i in range(self.n_init):

            m = self.set_model()

            m.fit([self.normalized_input], [self.y])
            log_likelihood.append(glm.loglikelihood_train)
            models.append(m)

        max_ll_loc = np.argmax(log_likelihood)
        model_max = models[max_ll_loc]

    #         #TODO: This has to move somewhere else!!
    # def fit_model(self,data_in:WheelData):
    #     model_info = {}
    #     transformer = GLMHMMTransfromer()
    #     transformer.transform_data(input_data=data_in.data)
    #     input_matrix, y, rewarded = transformer.get_session_unnormalized_data()

    #     #normalize and discard nogos
    #     normalized_input = np.copy(input_matrix)
    #     normalized_input[:,0] = preprocessing.scale(normalized_input[:,0])

    #     y = y.astype('int')
    #     go_idx = [i for i,row in enumerate(y) if row != -1]
    #     y = y[go_idx]
    #     normalized_input = normalized_input[go_idx]

    #     np.random.seed(65)
    #     C = 2
    #     n_init = 10
    #     M = normalized_input.shape[1]
    #     log_likelihood = []
    #     glms = []
    #     for i in range(n_init):
    #         glm = Glm(M,C)
    #         glm.fit([normalized_input],[y])
    #         log_likelihood.append(glm.loglikelihood_train)
    #         glms.append(glm)

    #     max_ll_loc = np.argmax(log_likelihood)
    #     glm_max = glms[max_ll_loc]

    #     model_info['loglikelihood_max'] = log_likelihood[max_ll_loc]
    #     model_info['weights'] = -glm_max.Wk[0][0]
    #     model_info['label_for_plot'] = ['stim','prev_choice','wsls','bias']
    #     self.model_info = model_info
