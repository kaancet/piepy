import os
import glob
import natsort
from os.path import join as pjoin
from os.path import exists as exists
from .config import config as cfg
from .exceptions import PathSettingError


class Paths:
    """Paths for single run"""
    def __init__(self, paths_dict: dict, path_idx: int = None) -> None:
        for name, path in paths_dict.items():
            if isinstance(path, list):
                if len(path) == 1:
                    setattr(self, name, path[0])
                else:
                    # TODO: MULTIPLE STIMLOG RIGLOG STITCHING HERE?
                    if path_idx is None:
                        raise PathSettingError(
                            f"Multiple runs exist in {path}, specify the run to get the path"
                        )
                    else:
                        # run no given
                        try:
                            ret = path[path_idx]
                            setattr(self, name, ret)
                        except IndexError:
                            raise IndexError(
                                f"Run no {path_idx} is larger than number of runs present in {name} location: {len(path)}"
                            )

            else:
                setattr(self, name, path)


class PathFinder:
    def __init__(self, sessiondir: str) -> None:
        self.sessiondir = sessiondir
        self.config_paths = cfg.paths
        self.init_directories_from_config()
        self.set_log_paths()

    def init_directories_from_config(self) -> None:
        """Initializes different experiment related paths from the config file"""
        tmp_runs = None
        self.all_paths = {}
        for name, path in self.config_paths.items():
            # multiple path options, try each one
            if name == "analysis":
                continue
            _found = False
            for p in path:
                if os.path.isdir(p):
                    _dirs = pjoin(p, self.sessiondir).replace("\\", os.sep)
                    _found_dirs = glob.glob(f"{_dirs}*")
                    if len(_found_dirs):
                        session_related_dir = _found_dirs[0]
                        _found = True
                        found_runs = self.look_for_runs(session_related_dir)
                        # saving everything as a list
                        if found_runs is None:
                            self.all_paths[name] = [session_related_dir]
                            # setattr(self, name, session_related_dir)
                        else:
                            if name in ["training", "presentation"]:
                                tmp_runs = found_runs
                            self.all_paths[name] = [
                                pjoin(session_related_dir, i) for i in found_runs
                            ]
                            # setattr(self, name, [pjoin(session_related_dir,i) for i in found_runs])
                        break
                else:
                    # if not dir, directly set attr
                    if exists(p):
                        _found = True
                        self.all_paths[name] = p
            if not _found:
                self.all_paths[name] = None

        # analysis is treated differently
        # to save in both J and backup or many more the designated places if desired
        if name == "analysis":
            # self.all_paths['analysis'] = []
            save_paths = []
            if tmp_runs is not None:
                for t in tmp_runs:
                    save_paths.append(
                        [
                            pjoin(pp, self.sessiondir, t)
                            for pp in self.config_paths["analysis"]
                        ]
                    )
                    # self.all_paths['analysis'].append([self.config['analysis']])
            else:
                # self.all_paths['analysis'] = [self.config['analysis']]
                save_paths = [
                    [pjoin(pp, self.sessiondir) for pp in self.config_paths["analysis"]]
                ]
            self.all_paths["save"] = save_paths

        if self.all_paths["presentation"] is None and self.all_paths["training"] is None:
            raise FileNotFoundError(
                f"{self.sessiondir} does not exist in neither presentation or training!!"
            )
        elif (
            self.all_paths["presentation"] is not None
            and self.all_paths["training"] is not None
        ):
            raise FileExistsError(
                f"{self.sessiondir} exists both in presentation and training!!"
            )

    @staticmethod
    def look_for_runs(dir_path: str) -> None:
        """Looks for existing run directories inside session directories,
        returns them as a list if present"""
        for root, dirs, files in os.walk(dir_path):
            if len(dirs) == 0:
                # this will happen if
                # there is only a single run in a session
                # and the logs are not in a dedicated run directory
                return None
            else:
                # there are dedicated run folders return them
                dirs = natsort.natsorted(dirs)
                return dirs

    def set_log_paths(self) -> None:
        """Sets the log paths as class attributes,
        even single runs are saved in lists(1 element)"""
        _to_add = {}
        for name, dir_path in self.all_paths.items():
            if dir_path is not None:
                if name in ["presentation", "training"]:
                    to_get = ["stimlog", "riglog", "prefs", "prot"]
                    for g in to_get:
                        temp_logs = self._get_(dir_path, g)
                        _to_add[g] = temp_logs
                        # setattr(self, g, temp_logs)
                elif name in ["onepcam", "eyecam", "facecam"]:
                    cam_logs = self._get_(dir_path, "camlog")
                    _to_add[f"{name}log"] = cam_logs
                    # setattr(self, f'{name}log', cam_logs)
                elif name == "save":
                    tmp_data = []
                    for i in dir_path:
                        tmp_data.append([pjoin(j, "runData.parquet") for j in i])
                    _to_add["data"] = tmp_data  # data is mostly to check existence
                    # setattr(self, 'data',[pjoin(i,'sessionData.parquet') for i in dir_path])
        self.all_paths = {**self.all_paths, **_to_add}

    @staticmethod
    def _get_(dir_path: list, to_get: str) -> list:
        """ """
        _logs = []
        if dir_path is not None:
            for i in dir_path:
                for l in os.listdir(i):
                    if l.endswith(to_get):
                        _logs.append(pjoin(i, l))
                        break
        return _logs
