import os
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime as dt
from ..utils import *
from .dbinterface import DataBaseInterface
from ..gsheet_functions import GSheet


class DataPaths:
    def __init__(self,sessiondir:str,runno=None):
        self.sessiondir = sessiondir
        self.sessionPath: str = None
        self.runno = runno

        self.init_from_config()
        if self.sessionPath is None:
            raise FileNotFoundError('Session directory {0} does not exist!'.format(self.sessionPath))
        self.get_log_paths()
        self.data = pjoin(self.savePath,'sessionData.csv').replace("\\",os.sep)

    def __repr__(self) -> str:
        kws = [f'{key}={value!r}' for key, value in self.__dict__.items()]
        return '{}({})'.format(type(self).__name__, ', '.join(kws))

    def init_from_config(self) -> None:
        self.config = getConfig()
        for k, v in self.config.items():
            session_path = pjoin(v,self.sessiondir).replace("\\",os.sep)
            setattr(self, k, session_path)
            # get where the raw data is either in training or presentation
            if 'presentation' in k or 'training' in k:
                if os.path.exists(session_path):
                    display(f'Found session rawdata at {v}')
                    self.sessionPath = session_path

            if 'analysis' in k:
                self.savePath = session_path

    def get_log_paths(self) -> None:
        for s_file in os.listdir(self.sessionPath):
            extension = os.path.splitext(s_file)[1]
            temp_key = extension.split('.')[-1]
            log_path = pjoin(self.sessionPath,s_file).replace("\\",os.sep)
            setattr(self, temp_key, log_path)
            if self.runno is not None:
                if self.runno in s_file:
                    log_path = pjoin(self.sessionPath,s_file).replace("\\","/")
                    setattr(self, temp_key, log_path)
            else:
                log_path = pjoin(self.sessionPath,s_file).replace("\\","/")
                setattr(self, temp_key, log_path)