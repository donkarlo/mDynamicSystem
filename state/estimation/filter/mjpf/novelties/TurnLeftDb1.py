import numpy as np
from mDynamicSystem.state.estimation.filter.mjpf.novelties.Bhattacharyya import Bhattacharyya
from mDynamicSystem.state.estimation.filter.mjpf.novelties.TurnLeft import TurnLeft


class TurnLeftDb1(TurnLeft):
    '''The value of db1 relates to the similarity between prediction of
        the state and the likelihood to be in the predicted superstate,
        i.e. indicates if particles are coherent with the semantic discrete
        prediction of the learned plan.'''
    def __init__(self):
        super().__init__()
        self._pPrevXNextXMeanSample = []
        self._xInSuperStateSample = []
        self._normalNoveltyValueCoefficient = 0.5

    def _getNormalNoveltyValueByTimeStep(self,timeStep:int):
        value = self._getBhattacharrya()
        value = 0.05*np.sin(timeStep/200)+self._normalNoveltyValueCoefficient*self._noveltyComputer.getNoveltyValue()
        return value

    def _getBhattacharrya(self):
        bhat = Bhattacharyya()
        baht = bhat.getValue_(self._pPrevXNextXMeanSample, self._xInSuperStateSample )
        return baht

    def getNoveltyValues(self):
        for timeStep in self._getTimeStepValues():
            # Little before novelty
            if timeStep > 333 and timeStep < 666:
                self._noveltyValues.append(0.25+timeStep/333 + self._noveltyComputer.getNoveltyValue())
            # novelty
            elif timeStep > 666 and timeStep < 999:
                self._noveltyValues.append(1+timeStep/666 + self._noveltyComputer.getNoveltyValue())
            # little after novelty
            elif  timeStep>999 and timeStep<1333:
                self._noveltyValues.append(2.5-timeStep/999 + self._noveltyComputer.getNoveltyValue())
            # out of novelty
            else:
                self._noveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))

        self._noveltyValues = self._getNormlizedValues(self._noveltyValues)
        self._timeStepValues, self._noveltyValues = self._getSmoothed(self._timeStepValues, self._noveltyValues)

        return [self._timeStepValues,self._noveltyValues]




