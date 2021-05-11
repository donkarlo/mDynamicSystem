from mMath.linearalgebra.Vector import Vector


class State(Vector):
    """"""

    def __init__(self,  time:int, refVec:Vector):
        """"""
        self.__time = time
        self.__refVec = refVec
