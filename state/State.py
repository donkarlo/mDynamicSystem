from mMath.data.timeSerie.stochasticProcess.markov.state.State import State as MainState
from mMath.linearalgebra.Vector import Vector


class State(MainState):
    """"""

    def __init__(self,  refVec:Vector):
        """"""
        self.__refVec = refVec
