from typing import List

from mDynamicSystem.state.observation import Observation
from mDynamicSystem.state import ObservationSerie


class ObservationSerieBuilder():
    def __init__(self):
        self.__observationList = List[Observation]

    def appendObservation(self, obs: Observation) -> None:
        self.__observationList.appendObservation(obs)

    def getObservationSerie(self) -> ObservationSerie:
        self.__obsSerie = ObservationSerie(self.__observationList)
        return self.__obsSerie
