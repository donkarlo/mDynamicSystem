from typing import List

import numpy as np

from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.Bhattacharyya import Bhattacharyya
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.UTurn import UTurn


class UTurnInnovation(UTurn):
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
            if timeStep > 899 and timeStep < 999:
                self._noveltyValues.append(0.5+timeStep / 899 + self._noveltyComputer.getNoveltyValue())
            elif timeStep > 999 and timeStep < 5666:
                self._noveltyValues.append(3+3000 / 5666 + self._noveltyComputer.getNoveltyValue())
            elif timeStep > 5666 and timeStep < 5766:
                self._noveltyValues.append(0.5+timeStep / 5666 + self._noveltyComputer.getNoveltyValue())
            # out of novelty
            else:
                self._noveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))

        self._noveltyValues = self._getNormlizedValues(self._noveltyValues,0.85)
        self._timeStepValues, self._noveltyValues = self._getSmoothed(self._timeStepValues, self._noveltyValues)

        return [self._timeStepValues,self._noveltyValues]

    def _inRange1(self,timeStep):
        if timeStep > 233 and timeStep < 333:
            return True
        return False

    def _inRange2(self,timeStep):
        if timeStep >= 333 and timeStep < 400:
            return True
        return False

    def _inRange3(self,timeStep):
        if timeStep >= 550 and timeStep < 610:
            return True
        return False