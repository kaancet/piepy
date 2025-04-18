from .run import RunMeta, Run
from .utils import timeit

from .pathfinder import Paths, PathFinder


class Session:
    def __init__(
        self, sessiondir: str, load_flag: bool = False, save_mat: bool = False
    ):
        """A base Session object, reads and aggregates the recorded data which can then be used in user specific
        analysis pipelines

        Args:
            sessiondir (str): directory of the session inside the presentation folder(e.g. 200619_KC033_wheel_KC)
            load_flag (bool, optional):  flag to either load previously parsed data or to parse it again. Defaults to False
            save_mat (bool, optional):   flag to make the parser also output a .mat file to be used in MATLAB scripts. Defaults to False

        """
        self.sessiondir = sessiondir
        self.load_flag = load_flag
        self.save_mat = save_mat
        self.runs = []
        self.metas = []
        self.stats = []

        # find relevant data paths
        self.paths = PathFinder(self.sessiondir)

        # look at run count
        self.run_count = len(self.paths.all_paths["stimlog"])

    def init_session_runs(self, skip_google: bool = True) -> None:
        """Initializes runs in a session, to be overwritten by other Session types(e.g. WheelDetectionSession)

        Args:
            skip_google (bool): Flag to skip parsing google sheets
        """
        for r in range(self.run_count):
            _path = Paths(self.paths.all_paths, r)
            self.runs.append(Run(_path))
            self.metas.append(RunMeta.get_meta(_path, skip_google))

    @timeit("Saving...")
    def save_session(self) -> None:
        """Saves the session data, meta and stats"""
        for run in self.runs:
            run.save_run(self.save_mat)

    @timeit("Loading...")
    def load_session(self) -> None:
        """Helper method to loop through the runs and load data and stats"""
        for run in self.runs:
            run.load_run()

    ####
    # DATABASE RELATED, NOT USED AT THE MOMENT
    ###

    # def save_to_db(self, db_dict: dict) -> None:
    #     """Checks if an entry for the session already exists and saves/updates accordingly"""
    #     if not self.db_interface.exists({"sessionId": self.meta.session_id}, "sessions"):
    #         self.db_interface.add_entry(db_dict, "sessions")
    #         self.db_interface.update_entry(
    #             {"id": self.meta.animalid},
    #             {"nSessions": self.current_session_no},
    #             "animals",
    #         )
    #     else:
    #         self.db_interface.update_entry(
    #             {"sessionId": self.meta.session_id}, db_dict, "sessions"
    #         )
    #         display(
    #             f"Session with id {self.meta.session_id} is already in database, updated the entry"
    #         )

    # def get_latest_trial_count(self):
    #     """Gets the last trial count from"""
    #     prev_trials = self.db_interface.get_entries({"id": self.meta.animalid}, "trials")
    #     try:
    #         return int(prev_trials["total_trial_no"].iloc[-1])
    #     except:
    #         return 0

    # def overall_session_no(self) -> int:
    #     """Gets the session number of the session"""
    #     mouse_entry = self.db_interface.get_entries(
    #         {"id": self.meta.animalid}, table_name="animals"
    #     )
    #     if len(mouse_entry):
    #         last_session_no = mouse_entry["nSessions"].iloc[0]
    #     else:
    #         display(f"No entry for mouse {self.meta.animalid} in animals table!")
    #         last_session_no = 0

    #     current_session_no = last_session_no + 1
    #     return current_session_no

    # def remove_session_db(self):
    #     """ """
    #     pass
