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


class StateMachineError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class LogTypeMissingError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FrameLoggingError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ScreenPulseError(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class VstimLoggingError(BaseException):
    def __init__(self, *args):
        super().__init__(*args)
