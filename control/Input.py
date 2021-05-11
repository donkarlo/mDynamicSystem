from mMath.linearalgebra.Vector import Vector


class Action(Vector):
    '''Each control is a vector that pushes system state tpward a new state.
    The simplest form is the derivative of velocity'''
    def __init__(self, time: float, refVec: Vector):
        self.__time = time
        self.__refVec = refVec