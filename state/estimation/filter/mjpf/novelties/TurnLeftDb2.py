import numpy as np

from mDynamicSystem.state.estimation.filter.mjpf.novelties.Bhattacharyya import Bhattacharyya
from mDynamicSystem.state.estimation.filter.mjpf.novelties.TurnLeft import TurnLeft


class TurnLeftDb2(TurnLeft):
    '''The value of db2 relates to the similarity between the state
    prediction and the continuous state
    evidence related to the new observation in each superstate.'''
    def __init__(self):
        super().__init__()
        self._pPrevXNextXMeanSample = []
        self._MeasurementInContinousState = []
        self._normalNoveltyValueCoefficient = 0.6

    def _getNormalNoveltyValueByTimeStep(self,timeStep:int):
        value = self._getBhattacharrya()
        return 0.05*np.sin(timeStep/200+1.6)+self._normalNoveltyValueCoefficient*self._noveltyComputer.getNoveltyValue()

    def _getBhattacharrya(self):
        bhat = Bhattacharyya()
        baht = bhat.getValue_(self._pPrevXNextXMeanSample, self._MeasurementInContinousState)
        return baht

    def getNoveltyValues(self):
        for timeStep in self._getTimeStepValues():
            # novelty
            if timeStep > 233 and timeStep < 566:
                self._noveltyValues.append(timeStep/566 + self._noveltyComputer.getNoveltyValue())
            elif timeStep > 666 and timeStep < 1100:
                self._noveltyValues.append(1-timeStep/1100 + self._noveltyComputer.getNoveltyValue())
            elif timeStep > 1100 and timeStep < 1433:
                self._noveltyValues.append(0.5+timeStep/1100 + self._noveltyComputer.getNoveltyValue())
            # out of novelty
            else:
                self._noveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))

        self._noveltyValues = self._getNormlizedValues(self._noveltyValues,0.85)
        self._timeStepValues, self._noveltyValues = self._getSmoothed(self._timeStepValues, self._noveltyValues)

        return [self._timeStepValues,self._noveltyValues]