import sys
import logging
from logging.handlers import QueueHandler
import traceback
from os.path import join as pjoin
from .io import display


class Logger:
    def __init__(self, log_path: str) -> None:
        self.log_path = pjoin(log_path, "analysis_log.log")
        self.prefix = ""
        self.logger = self.listener_configurer()

    def set_msg_prefix(self, prefix: str) -> None:
        self.prefix = prefix

    def _log(self, level, msg: str, prefix: str = None, cml: bool = False) -> None:
        if prefix is None:
            prefix = self.prefix
        logger = logging.getLogger()
        msg = f"{self.prefix} - {msg}"
        logger.log(level, msg)
        if cml:
            print(msg)

    def info(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        """INFO Logging"""
        self._log(logging.INFO, msg, prefix, cml)

    def warning(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        """WARNING Logging"""
        self._log(logging.WARNING, msg, prefix, cml)

    def error(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        """ERROR Logging"""
        self._log(logging.ERROR, msg, prefix, cml)

    def critical(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        """CRITICAL Logging"""
        self._log(logging.CRITICAL, msg, prefix, cml)

    def debug(self, msg: str, prefix: str = None, cml: bool = False) -> None:
        """DEBUG Logging"""
        self._log(logging.DEBUG, msg, prefix, cml)

    def listener_configurer(self):
        """Configures the logger
        Returns:
            logger: configured logging object
        """
        logger = logging.getLogger()

        fh = logging.FileHandler(pjoin(self.log_path), mode="w", encoding="utf-8")
        fmtr = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
        fh.setFormatter(fmtr)
        logger.setLevel(logging.INFO)
        logger.addHandler(fh)

    def listener_process(self, queue) -> None:
        """Listener process is a target for a multiprocess process
        that runs and listens to a queue for logging events.

        Arguments:
            queue (multiprocessing.manager.Queue): queue to monitor
            configurer (func): configures loggers
            log_name (str): name of the log to use

        Returns:
            None
        """
        self.listener_configurer()

        while True:
            try:
                record = queue.get()
                if record is None:
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)
            except Exception:
                print("Failure in listener_process", file=sys.stderr)
                traceback.print_last(limit=1, file=sys.stderr)

    def worker_configurer(self, queue) -> None:
        h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        # send all messages, for demo; no other level or filter logic applied.
        root.setLevel(logging.INFO)
