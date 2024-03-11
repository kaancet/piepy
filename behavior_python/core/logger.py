import os
import logging
from os.path import join as pjoin
from ..utils import display


class Logger:
    def __init__(self,
                 log_path:str) -> None:
        self.log_path = pjoin(log_path,'analysis_log.log') 
        
        logging.basicConfig(level=logging.INFO,
                            filename=self.log_path,
                            filemode='w', #'a' if append else 'w',
                            encoding='utf-8',
                            format="%(asctime)s : %(levelname)s : %(message)s")
        
        # session_dir = log_path.split(sep=os.sep)[-1]
        # init_msg = f"Started analysis of {session_dir}"
        self.prefix = ''
        # self.info(init_msg,cml=True)
    
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