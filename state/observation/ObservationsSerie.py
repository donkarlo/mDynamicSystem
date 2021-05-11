from typing import List

from mDynamicSystem.state.observation.Observation import Observation


class ObservationsSerie():
    """A time serie of observation vectors
    """

    def __init__(self, observationList:List[Observation]):
        ''''''
        self.__observationList = observationList

    def getByIdx(self, index: int) -> int:
        ''''''
        return self.__observationList[index]

    def getObservationList(self)->List[Observation]:
        ''''''
        return self.__observationList

    def getLastObservation(self)->Observation:
        ''''''
        return self.__observationSerie.getByIdx(self.getLastObservationIndex())

    def getObservationSlice(self,startIndex,endIndex)->List[Observation]:
        ''''''
        return self.__observationList[startIndex:endIndex]

    def getLastObservationIndex(self):
        return len(self.__observationList)-1
