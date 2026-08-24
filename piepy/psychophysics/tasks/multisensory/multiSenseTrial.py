from ....core.trial import Trial


class MultiSenseTrial(Trial):
    def __init__(self, trial_no: int, meta, logger) -> None:
        super().__init__(trial_no, meta, logger)
