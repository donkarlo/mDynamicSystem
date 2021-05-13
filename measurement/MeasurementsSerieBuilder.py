from typing import List

from mDynamicSystem.measurement import Measurement
from mDynamicSystem.state import ObservationSerie


class MeasurementsSerieBuilder():
    def __init__(self):
        self.__observationList = List[Measurement]

    def appendMeasurement(self, obs: Measurement) -> None:
        self.__observationList.append(obs)

    def getObservationSerie(self) -> ObservationSerie:
        self.__obsSerie = ObservationSerie(self.__observationList)
        return self.__obsSerie
