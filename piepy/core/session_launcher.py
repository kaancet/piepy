import argparse
import importlib
from os.path import join as pjoin
from os.path import dirname, abspath, normpath, exists


def main():
    parser = argparse.ArgumentParser(description="Session Data Parsing Tool")

    parser.add_argument(
        "sessiondir",
        metavar="sessiondir",
        type=str,
        help="Directry of the session, e.g. 231103_KC143_detect__no_cam_KC",
    )
    parser.add_argument(
        "-p",
        "--paradigm",
        metavar="paradigm",
        type=str,
        help="Behavior paradigm(e.g. detection)",
    )
    parser.add_argument(
        "-l",
        "--load",
        default=True,
        metavar="load_type",
        action=argparse.BooleanOptionalAction,
        type=str,
        help="Loading parameter",
    )

    """
    session -p detection --load 231103_KC143_detect__no_cam_KC
    """

    opts = parser.parse_args()

    session_class_name = f"{opts.paradigm}Session"
    if opts.paradigm == "detection":
        session_class_name = session_class_name[0].upper() + session_class_name[1:]
        session_class_name = f"wheel{session_class_name}"

    # __file__ is core.session_launcher.py
    mod_path = normpath(
        pjoin(
            abspath(dirname(dirname(__file__))),
            "psychophysics",
            opts.paradigm,
            session_class_name + ".py",
        )
    )
    if exists(mod_path):
        mod = importlib.import_module(
            f"piepy.psychophysics.{opts.paradigm}.{session_class_name}"
        )
        session_class_name = (
            session_class_name[0].upper() + session_class_name[1:]
        )  # uppercasing the first letter for class name
        session_parser = getattr(mod, session_class_name)
    else:
        raise ModuleNotFoundError(f"No module found at {mod_path}")

    if opts.load:
        skip_google = True
    else:
        skip_google = False

    session_parser(
        sessiondir=opts.sessiondir, load_flag=opts.load, skip_google=skip_google
    )


if __name__ == "__main__":
    main()
