from mMath.data.timeSerie.stochasticProcess.state.State import State as MainState
from mMath.linearAlgebra.Vector import Vector


class State(MainState):
    """"""

    def __init__(self,  refVec:Vector):
        '''

        :param refVec:
        '''
        super().__init__()
        self.__refVec:Vector = refVec

    def getRefVec(self)->Vector:
        '''

        :return:
        '''
        return self.__refVec
