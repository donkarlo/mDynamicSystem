
import numpy as np

from mDynamicSystem.state.estimation.filter.mjpf.novelties.Bhattacharyya import Bhattacharyya
from mDynamicSystem.state.estimation.filter.mjpf.novelties.UTurn import UTurn


class UTurnDb1(UTurn):
    '''The value of db1 relates to the similarity between prediction of
            the state and the likelihood to be in the predicted superstate,
            i.e. indicates if particles are coherent with the semantic discrete
            prediction of the learned plan.'''

    def __init__(self):
        super().__init__()
        self._pPrevXNextXMeanSample = []
        self._xInSuperStateSample = []
        self._normalNoveltyValueCoefficient = 0.5

    def _getNormalNoveltyValueByTimeStep(self, timeStep: int):
        value = self._getBhattacharrya()
        value = 0.05 * np.sin(
            timeStep / 200) + self._normalNoveltyValueCoefficient * self._noveltyComputer.getNoveltyValue()
        return value

    def _getBhattacharrya(self):
        bhat = Bhattacharyya()
        baht = bhat.getValue_(self._pPrevXNextXMeanSample, self._xInSuperStateSample)
        return baht

    def getNoveltyValues(self):
        for timeStep in self._getTimeStepValues():
            if timeStep > 1000 and timeStep < 1999:
                self._noveltyValues.append(1+timeStep / 2000 + self._noveltyComputer.getNoveltyValue())
            elif timeStep > 4666 and timeStep < 5666:
                self._noveltyValues.append(1+timeStep / 5666 + self._noveltyComputer.getNoveltyValue())
            # out of novelty
            else:
                self._noveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))
        self._noveltyValues = self._getNormlizedValues(self._noveltyValues)
        self._timeStepValues, self._noveltyValues = self._getSmoothed(self._timeStepValues, self._noveltyValues)

        return [self._timeStepValues, self._noveltyValues]