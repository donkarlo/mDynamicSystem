from mMath.data.timeSerie.stochasticProcess.state.State import State as StochasticProcessState
from mMath.linearAlgebra.Vector import Vector


class State(StochasticProcessState):
    def __init__(self,center:Vector,standardDeviation:Vector):
        '''

        :param center:
        :param standardDeviation:
        :param standardDerviationCoefficient:
        '''
        self._center:Vector = center
        self._standardDeviation:Vector = standardDeviation

    def getCenter(self)->Vector:
        return self._center

    def getStandardDeviation(self)->Vector:
        return self._standardDeviation
