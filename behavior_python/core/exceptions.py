from behavior_python.utils import display


class WrongSessionTypeError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        