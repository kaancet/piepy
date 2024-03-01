import os
import logging
from os.path import join as pjoin
from ..utils import *


class DataPaths:
    def __init__(self, sessiondir: str):
        self.sessiondir = sessiondir
        self.sessionPath: str = None

        self.init_from_config()
        if self.sessionPath is None:
            raise FileNotFoundError(
                "Session directory {0} does not exist!".format(self.sessionPath)
            )
        self.get_log_paths()
        # self.data = pjoin(self.savePath,'sessionData.csv').replace("\\",os.sep)
        self.dataPaths = []
        for r in self.runPaths:
            self.dataPaths.append(
                pjoin(self.savePath, "sessionData.parquet").replace("\\", os.sep)
            )
        if len(self.dataPaths) == 1:
            self.data = self.dataPaths[0]

    def __repr__(self) -> str:
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def init_from_config(self) -> None:
        """Initializes different experiment related paths from the config file"""
        self.config = getConfig()
        for k, v in self.config.items():
            session_path = pjoin(v, self.sessiondir).replace("\\", os.sep)
            if os.path.exists(session_path):
                display(f"Found session related recordings at {v}")
                if "presentation" in k or "training" in k:
                    # set the session path as the presentation directory
                    self.sessionPath = session_path

            if "analysis" in k:
                self.savePath = session_path
            setattr(self, k, session_path)

    def get_log_paths(self) -> None:
        self.runPaths = []
        for root, _, files in os.walk(self.sessionPath):
            r_paths = {"runname": root.split(os.sep)[-1]}
            for f in files:
                s_file = pjoin(root, f)
                extension = os.path.splitext(s_file)[1]
                temp_key = extension.split(".")[-1]
                log_path = s_file.replace("\\", os.sep)
                r_paths[temp_key] = log_path
            if len(r_paths) > 1:
                self.runPaths.append(r_paths)

        if len(self.runPaths) == 1:
            # there is only one run in a session, set class attributes to those
            for k, v in self.runPaths[0].items():
                setattr(self, k, v)


class Logger:
    def __init__(self, log_path: str, append: bool) -> None:
        self.log_path = pjoin(log_path, "analysis_log.log")

        logging.basicConfig(
            level=logging.INFO,
            filename=self.log_path,
            filemode="a" if append else "w",
            encoding="utf-8",
            format="%(asctime)s : %(levelname)s : %(message)s",
        )

        session_dir = log_path.split(sep=os.sep)[-1]
        init_msg = f"Started analysis of {session_dir}"
        self.prefix = ""
        self.info(init_msg, cml=True)

    def set_msg_prefix(self, prefix: str) -> None:
        self.prefix = prefix

    def reset_prefix(self) -> None:
        self.prefix = ""

    def debug(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f"{prefix.upper()} : {msg}"

        if cml:
            display(msg)
        logging.debug(msg)

    def info(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f"{prefix.upper()} : {msg}"

        if cml:
            display(msg)
        logging.info(msg)

    def warning(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f"{prefix.upper()} : {msg}"

        if cml:
            display(msg)
        logging.warning(msg)

    def error(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f"{prefix.upper()} : {msg}"

        if cml:
            display(msg)
        logging.error(msg)

    def critical(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f"{prefix.upper()} : {msg}"

        if cml:
            display(msg)
        logging.critical(msg)
