from mMath.linearAlgebra.Vector import Vector


class Input(Vector):
    '''Each control is a vector that pushes system state tpward a new state.
    The simplest form is the derivative of velocity'''
    def __init__(self, refVec: Vector,time: float):
        self.__time = time
        self.__refVec = refVec