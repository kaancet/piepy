import os
import json
from pathlib import Path

# Prefilled JSON content
CONFIG = {
    "multiprocess": {"enable": False, "cores": 8},
    "verbose": True,
    "paths": {"colors": [str(Path.cwd().parents[0] / "plotters" / "colors")]},
}

PATHS = ["presentation", "training", "twop", "onepcam", "facecam", "eyecam", "opto_pattern", "analysis"]


class Config:
    def __init__(self) -> None:

        self.home_dir = Path.home()
        self.config_dir = self.home_dir / ".piepy"
        self.file_path = self.config_dir / "config.json"

        # Check if file already exists
        if self.file_path.exists():
            with open(self.file_path) as json_file:
                conf = json.load(json_file)
        else:
            conf = self.make_config()

        for k, v in conf.items():
            setattr(self, k, v)

    def _set(self, name: str, value) -> None:
        """Function to set attributes"""
        setattr(self, name, value)

    ## some methods for on the fly changes to config file
    def set_verbosity(self, val: bool) -> None:
        """Set verbosity"""
        if not isinstance(val, bool):
            raise ValueError(f"Please provie a boolean to set verbosity, got {type(val)}")

        self._set("verbose", val)

    def make_config(self):
        """Creates a .piepy folder at the home directory and puts config.json in there"""
        print(f"Creating .piepy directory at {self.home_dir}", flush=True)
        os.makedirs(self.config_dir, exist_ok=True)

        # add the paths variable
        config = CONFIG.copy()

        for p in PATHS:
            config["paths"][p] = [f"data/{p}"]

        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        return config


config = Config()
