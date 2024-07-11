class WrongSessionTypeError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PrefProtMismatchError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NoRigReactionTimeError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class PathSettingError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
