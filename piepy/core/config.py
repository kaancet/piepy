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


config = Config()
