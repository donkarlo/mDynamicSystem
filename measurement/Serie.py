from typing import List

from mDynamicSystem.measurement.Measurement import Measurement


class Serie():
    """A time serie of measurement vectors
    """

    def __init__(self, measurements:List[Measurement]=None):
        ''''''
        self._measurementList = measurements if measurements is not None else []

    def getByMeasurementByIndex(self, index: int) -> int:
        ''''''
        return self._measurementList[index]

    def getMeasurementList(self)->List[Measurement]:
        ''''''
        return self._measurementList

    def getLastMeasurement(self)->Measurement:
        ''''''
        return self.getByMeasurementByIndex(self.getLastMeasurementIndex())

    def getFirstMeasurement(self):
        ''''''
        return self.getByMeasurementByIndex(0)

    def getMeasurementSlice(self, startIndex, endIndex)->List[Measurement]:
        ''''''
        return self._measurementList[startIndex:endIndex]

    def getLastMeasurementIndex(self):
        ''''''
        return len(self._measurementList) - 1

    def getLength(self)->int:
        return len(self._measurementList)

    def appendMeasurement(self, measurement: Measurement) -> None:
        self._measurementList.append(measurement)
