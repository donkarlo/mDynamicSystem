from mMath.data.timeSerie.stochasticProcess.state.State import State as MainState
from mMath.linearAlgebra.Vector import Vector


class State(MainState):
    """"""

    def __init__(self,  refVec:Vector):
        """"""
        self.__refVec = refVec
