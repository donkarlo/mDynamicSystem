from typing import List

from mDynamicSystem.measurement.Measurement import Measurement


class MeasurementsSerie():
    """A time serie of measurement vectors
    """

    def __init__(self, observationList:List[Measurement]):
        ''''''
        self.__measurements = observationList

    def getByIdx(self, index: int) -> int:
        ''''''
        return self.__measurements[index]

    def getMeasurements(self)->List[Measurement]:
        ''''''
        return self.__measurements

    def getLastMeasurement(self)->Measurement:
        ''''''
        return self.__measurements.getByIdx(self.getLastObservationIndex())

    def getObservationSlice(self,startIndex,endIndex)->List[Measurement]:
        ''''''
        return self.__measurements[startIndex:endIndex]

    def getLastObservationIndex(self):
        ''''''
        return len(self.__measurements) - 1
