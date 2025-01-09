import sys
import json
import numpy as np
from colorama import Fore, Style
from datetime import datetime as dt


from .config import config


def display(*msgs: str, color: str = "white", timestamp: bool = True):
    if config.verbose:
        try:
            fg_color = getattr(Fore, color.upper())
        except AttributeError:
            fg_color = Fore.WHITE
        msg = fg_color
        if timestamp:
            msg += f"[{dt.today().strftime('%y-%m-%d %H:%M:%S')}] - "
        msg += f"{''.join(msgs)}\n"
        msg += Style.RESET_ALL
        sys.stdout.write(msg)
        sys.stdout.flush()


def JSONConverter(obj):
    if isinstance(obj, dt):
        return obj.__str__()
    if isinstance(obj, np.ndarray):
        return obj.tolist()


def jsonify(data):
    """Jsonifies the numpy arrays inside the analysis dictionary, mostly for saving and pretty printing"""
    jsonified = {}

    for key, value in data.items():
        if isinstance(value, list):
            value = [jsonify(item) if isinstance(item, dict) else item for item in value]
        if isinstance(value, dict):
            value = jsonify(value)
        if type(value).__module__ == "numpy":
            value = value.tolist()
        jsonified[key] = value

    return jsonified


def save_dict_json(path: str, dict_in: dict) -> None:
    """Saves a dictionary as a .json file"""
    with open(path, "w") as fp:
        jsonstr = json.dumps(dict_in, indent=4, default=JSONConverter)
        fp.write(jsonstr)


def load_json_dict(path: str) -> dict:
    """Loads .json file as a dict
    :param path : path of the .json file
    :type path  : path string
    """
    with open(path) as f_in:
        return json.load(f_in)
