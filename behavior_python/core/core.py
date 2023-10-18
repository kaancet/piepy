import os
import logging
from os.path import join as pjoin
from ..utils import *


class DataPaths:
    def __init__(self,sessiondir:str,runno=None):
        self.sessiondir = sessiondir
        self.sessionPath: str = None
        self.runno = runno

        self.init_from_config()
        if self.sessionPath is None:
            raise FileNotFoundError('Session directory {0} does not exist!'.format(self.sessionPath))
        self.get_log_paths()
        # self.data = pjoin(self.savePath,'sessionData.csv').replace("\\",os.sep)
        self.data = pjoin(self.savePath,'sessionData.parquet').replace("\\",os.sep)

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
            if self.runno is not None:
                if self.runno in s_file:
                    log_path = pjoin(self.sessionPath,s_file).replace("\\","/")
                    setattr(self, temp_key, log_path)
            else:
                log_path = pjoin(self.sessionPath,s_file).replace("\\","/")
                if hasattr(self,temp_key):
                    # if there are multiple runs of the same session(this should be very rare)
                    _attr = getattr(self,temp_key)
                    if not isinstance(_attr,list):
                        _attr = [_attr]
                    _attr.append(log_path)
                    setattr(self,temp_key,_attr)
                else:
                    setattr(self, temp_key, log_path)
                    

class Logger:
    def __init__(self,
                 log_path:str,
                 append:bool) -> None:
        self.log_path = pjoin(log_path,'analysis_log.log') 
        
        logging.basicConfig(level=logging.INFO,
                            filename=self.log_path,
                            filemode='a' if append else 'w',
                            encoding='utf-8',
                            format="%(asctime)s : %(levelname)s : %(message)s")
        
        session_dir = log_path.split(sep=os.sep)[-1]
        init_msg = f"Started analysis of {session_dir}"
        self.prefix = ''
        self.info(init_msg,cml=True)
    
    def set_msg_prefix(self,prefix:str) -> None:
        self.prefix = prefix
        
    def reset_prefix(self) -> None:
        self.prefix = ''

    def debug(self,msg:str,prefix:str=None,cml:bool=False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f'{prefix.upper()} : {msg}'
        
        if cml:
            display(msg)
        logging.debug(msg)
    
    def info(self,msg:str,prefix:str=None,cml:bool=False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f'{prefix.upper()} : {msg}'
            
        if cml:
            display(msg)
        logging.info(msg)
    
    def warning(self,msg:str,prefix:str=None,cml:bool=False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f'{prefix.upper()} : {msg}'
            
        if cml:
            display(msg)
        logging.warning(msg)
    
    def error(self,msg:str,prefix:str=None,cml:bool=False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f'{prefix.upper()} : {msg}'
            
        if cml:
            display(msg)
        logging.error(msg)
        
    def critical(self,msg:str,prefix:str=None,cml:bool=False) -> None:
        if prefix is None:
            prefix = self.prefix
        msg = f'{prefix.upper()} : {msg}'
            
        if cml:
            display(msg)
        logging.critical(msg)