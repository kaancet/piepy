from .cameraAnalysis import *


class FaceCamAnalysis(CamDataAnalysis):
    def __init__(self, runpath: str, data: DataFrame) -> None:
        super().__init__(runpath, data)