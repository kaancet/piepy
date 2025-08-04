import os
import natsort
import hashlib
import importlib
import polars as pl
from types import ModuleType
from datetime import datetime as dt
from multiprocessing import Pool, set_start_method
from os.path import dirname, abspath, normpath, join

from .io import display
from .utils import clean_string
from .config import config as cfg


class TaskHub:
    def __init__(self, paradigm: str):
        """_summary_

        Args:
            paradigm (str): The paradigm that will be used to
        """
        self.set_paradigm(paradigm)

    @staticmethod
    def generate_unique_session_id(
        baredate: str, animalid: str, *args, digit_len: int = 7
    ) -> int:
        """Generates a unique run id
        NOTE: this assumes baredate and animalid combinations are unique

        Args:
            baredate (str): date in string form (e.g. 240801)
            animalid (str): ID of the animal in string form (e.g. KC133)
            digit_len (int, optional): length of digits in the unique ID. Defaults to 6.

        Returns:
            int: unique id
        """
        all_strings = [baredate, animalid, *args]
        combined_string = "|".join(sorted(all_strings))
        # Create a hash from the combined string
        hash_object = hashlib.sha256(combined_string.encode("utf-8"))
        # Convert the hash to an integer
        unique_id = int(hash_object.hexdigest(), 16) % 10**digit_len
        return unique_id

    @staticmethod
    def _get_session_handler(paradigm: str) -> ModuleType:
        """Initializes the relevant session analyzer

        Args:
            paradigm (str): Name of the paradigm, sperated with underscores for each diectory in psychophysics
            For example, analysis of wheel detection task is done by the wheelDetectionSession.py,
            which is at the location psychophysics/wheel/detection so the paradigm argument will be wheel_detection

        Raises:
            ModuleNotFoundError: Module not found

        Returns:
            ModuleType: The module that can handle the analysis of given task paradigm
        """

        if "_" in paradigm:
            # each underscore corresponds to a subcategory,
            # the last one being where the class resides in
            # e.g. wheel_detection will be in wheel/detection
            paradigm_dirs = paradigm.split("_")

        paradigm_file = f"{''.join([p.capitalize() if i != 0 else p for i, p in enumerate(paradigm_dirs)])}Hub.py"
        paradigm_class = paradigm_file[0].upper() + paradigm_file[1:].strip(".py")

        paradigm_path = normpath(
            join(
                abspath(dirname(dirname(__file__))),  # __file__ is core.mouse.py
                "psychophysics",  # this is for behavioral analysis classes
                os.sep.join(paradigm_dirs),
                paradigm_file,
            )
        )
        if os.path.exists(paradigm_path):
            _dirs = paradigm_path.split(os.sep)
            _indx = _dirs.index("piepy")  # find the first occurence of piepy

            mod = importlib.import_module(
                ".".join(_dirs[_indx + 1 :])[:-3]  # remove the ".py" at the end
            )
            return getattr(mod, paradigm_class)
        else:
            raise ModuleNotFoundError(f"No module found at {paradigm_path}")

    def set_paradigm(self, paradigm: str) -> None:
        """Sets the paradigm of which the read sessions will be analysed in

        Args:
            paradigm (str): Name of the paradigm, eg. wheel_detection
        """
        self.paradigm = clean_string(paradigm)

        # set session parser
        self.session_handler = self._get_session_handler(self.paradigm)
        display(f"Set the data analysis paradigm to {self.paradigm}", color="cyan")

    def initialize(
        self,
        data: pl.DataFrame | list,
        load_sessions: str = False,
    ) -> None:
        """Initializes the TaskHub using either a list of sessions or a previously analyzed dataFrame

        Args:
            data (pl.DataFrame | list): Either a previously analytzed and saved dataFrame or the sessions list
            load_sessions (str, optional): Whether to load the sessons when using the session list to initialize the TaskHub. Defaults to False.
        """
        if isinstance(data, pl.DataFrame):
            # if DataFrame is given, directly set it as data and generate an exp_lis
            self.data = data
            # to prevent discrepancy between user input paradigm and paradigm column in DataFrame filter for paradigm
            self.data = self.data.filter(pl.col("paradigm") == self.paradigm)
            if self.data.is_empty():
                display(
                    f">>> WARNING <<< No sessions/trials fit the {self.paradigm} paradigm, data is empty!",
                    color="red",
                )
            # try to extract experiment list from the data
            self.session_list = (
                self.data["session_path"].unique(maintain_order=True).to_list()
            )
        elif isinstance(data, list):
            filtered_list = self._filter_session_list(data)
            self.session_list = natsort.natsorted(filtered_list)

            self.gather_sessions(
                self.session_list,
                load_sessions=load_sessions,
            )

    def _filter_session_list(self, session_list: list) -> list:
        """A dedicated function depending on the task paradigm.
        Should be overwritten in dedicated hub classes

        Args:
            session_list (list): A list of sessions

        Returns:
            list: Filtered list of sessions
        """
        pass

    def _get_session(self, session_path: str) -> pl.DataFrame:
        """Method to analyze a single session
        Should be overwritten in dedicated hub classes

        Args:
            session_path (str): Path of the session

        Returns:
            pl.DataFrame: The session data. It is empty if there was an error during session analysis
        """
        pass

    def gather_sessions(
        self, session_list: list, load_sessions: bool = False
    ) -> pl.DataFrame | None:
        """Gathers all the sessions (uses paralell processing)

        Args:
            session_list(list):
            get_func(callable): The function that will create a session dataframe from a single run, using the session_handler
            load_session (bool, optional): Whether to load sessions or reanalyze them. Defaults to False.

        Returns:
            pl.DataFrame | None: DataFrame of all the trials in all the analysed sessions. None if no paradigm is not set
        """
        if self.paradigm is None:
            print(
                "Can't gather sessions without setting the paradigm first! Use set_paradigm to do that"
            )
            return None

        self.load_flag = load_sessions
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        with Pool(processes=cfg.multiprocess["cores"]) as pool:
            _list_data = pool.map(self._get_session, session_list)

        # concat everything on the list
        data = _list_data[0]
        for df2 in _list_data[1:]:
            if not len(df2.columns):
                continue
            df2 = df2.select(data.columns)
            if data.dtypes != df2.dtypes:
                data = data.with_columns(
                    [
                        pl.col(n).cast(t)
                        for n, t in zip(data.columns, df2.dtypes)
                        if t != pl.Null
                    ]
                )

            data = pl.concat([data, df2])

        data = data.sort(
            ["date", "animalid", "run_no"],
            descending=[False, False, False],
        )

        # make reorder column list
        reorder = ["total_trial_no"] + data.columns
        # in the end add a final all trial count column
        data = data.with_columns(
            pl.Series(name="total_trial_no", values=list(range(1, len(data) + 1)))
        )
        # reorder
        self.data = data.select(reorder)
        return self.data

    def save(self, saveloc: str = None) -> None:
        """Saves the parsed data

        Args:
            saveloc (str, optional): Save location. If not provided, will save into the directory of the latest session in the data
        """
        # make the name : <first_sessions_date>-<last_session_date>_<animal_list>_<analysis_date>.parquet
        date_first = self.data[0, "baredate"]
        date_last = self.data[-1, "baredate"]
        animal_list = self.data["animalid"].unique().sort().to_list()
        animal_list = ",".join(animal_list)

        savename = f"{date_first}-{date_last}_{animal_list}_{dt.strftime(dt.today(), '%y%m%d')}.parquet"
        if saveloc is None:
            # if no saveloc given, use the latest session as the save location
            saveloc = self.data[-1, "session_path"].replace("presentation", "analysis")

        self.data.write_parquet(f"{saveloc}/{savename}")
        display(f"Saved at {saveloc}", color="green")
