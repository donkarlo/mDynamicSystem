from typing import List

from mDynamicSystem.measurement.Measurement import Measurement


class MeasurementSerie():
    """A time serie of measurement vectors
    """

    def __init__(self, measurement:List[Measurement]):
        ''''''
        self.__measurements = measurement

    def getByMeasurementByIndex(self, index: int) -> int:
        ''''''
        return self.__measurements[index]

    def getMeasurements(self)->List[Measurement]:
        ''''''
        return self.__measurements

    def getLastMeasurement(self)->Measurement:
        ''''''
        return self.getByMeasurementByIndex(self.getLastMeasurementIndex())

    def getMeasurementSlice(self, startIndex, endIndex)->List[Measurement]:
        ''''''
        return self.__measurements[startIndex:endIndex]

    def getLastMeasurementIndex(self):
        ''''''
        return len(self.__measurements) - 1

    def getLength(self):
        return len(self.__measurements)
