import os
import json


class Config:
    def __init__(self) -> None:
        config_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with open(os.path.join(config_dir, "config.json")) as json_file:
            conf = json.load(json_file)

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


config = Config()
