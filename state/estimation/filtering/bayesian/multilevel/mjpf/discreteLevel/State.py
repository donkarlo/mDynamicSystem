from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.linearAlgebra.Vector import Vector


class State(State):
    def __init__(self,center:Vector,standardDeviation:Vector):
        '''

        :param center:
        :param standardDeviation:
        :param standardDerviationCoefficient:
        '''
        super().__init__(center)
        self._center:Vector = center
        self._standardDeviation:Vector = standardDeviation

    def getCenter(self)->Vector:
        return self._center

    def getStandardDeviation(self)->Vector:
        return self._standardDeviation
