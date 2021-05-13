from typing import List

from mDynamicSystem.measurement.Measurement import Measurement


class MeasurementsSerie():
    """A time serie of measurement vectors
    """

    def __init__(self, observationList:List[Measurement]):
        ''''''
        self.__observationList = observationList

    def getByIdx(self, index: int) -> int:
        ''''''
        return self.__observationList[index]

    def getMeasurements(self)->List[Measurement]:
        ''''''
        return self.__observationList

    def getLastObservation(self)->Measurement:
        ''''''
        return self.__observationSerie.getByIdx(self.getLastObservationIndex())

    def getObservationSlice(self,startIndex,endIndex)->List[Measurement]:
        ''''''
        return self.__observationList[startIndex:endIndex]

    def getLastObservationIndex(self):
        return len(self.__observationList)-1
