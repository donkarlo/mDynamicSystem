from typing import List

from mDynamicSystem.obs.Obs import Obs


class Serie():
    """A time serie of obs vectors
    """

    def __init__(self, measurements:List[Obs]=None):
        ''''''
        self._measurementList:List[Obs] = measurements if measurements is not None else []

    def getMeasurementByIndex(self, index: int) -> Obs:
        ''''''
        return self._measurementList[index]

    def getMeasurementList(self)->List[Obs]:
        ''''''
        return self._measurementList

    def getLastMeasurement(self)->Obs:
        ''''''
        return self.getMeasurementByIndex(self.getLastMeasurementIndex())

    def getFirstMeasurement(self):
        ''''''
        return self.getMeasurementByIndex(0)

    def getMeasurementSlice(self, startIndex, endIndex)->List[Obs]:
        ''''''
        return self._measurementList[startIndex:endIndex]

    def getMeasuremnentListUntilLastState(self):
        return self._measurementList[1:self.getLastMeasurementIndex()-1]

    def getLastMeasurementIndex(self):
        ''''''
        return len(self._measurementList) - 1

    def getLength(self)->int:
        return len(self._measurementList)

    def appendMeasurement(self, measurement: Obs) -> None:
        self._measurementList.append(measurement)
