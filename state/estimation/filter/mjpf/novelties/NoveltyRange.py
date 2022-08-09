from typing import List

from mDynamicSystem.state.estimation.filter.mjpf.novelties.NoveltyPattern import NoveltyPattern


class NoveltyRange:
    def __init__(self
                 ,xRangeStart:int
                 ,xRangeEnd:int
                 ,xRangeStep
                 ,nomalityRate:float
                 ,noveltyPeaks:List[NoveltyPattern]
                 ):
        self._novetyPeaks = noveltyPeaks

    def getNoveltySignal(self):
        for noveltyPeak in self._novetyPeaks:
            pass

    def getNormlizedValues(self, values: List[float], coefficient: float = 1):
        maxValue = max(values)
        minValue = min(values)
        for yValueCounter, value in enumerate(values):
            values[yValueCounter] = coefficient * (value - minValue) / (maxValue - minValue)
        return values

