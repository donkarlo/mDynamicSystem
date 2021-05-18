from mMath.linearAlgebra.Vector import Vector


class Pose(Vector):
    '''Pose is considering the system without any control input'''
    def __init__(self, time: float, refVec: Vector):
        self.__time = time
        self.__refVec = refVec