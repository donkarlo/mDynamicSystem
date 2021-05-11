import numpy as np

from mDynamicSystem.state import Observation


class ThreeDPosVelObs(Observation):
    def __init__(self, time: float, x: float, y: float, z: float, vx: float, vy: float, vz: float):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__vx = vx
        self.__vy = vy
        self.__vz = vz
        obsVector = np.array([self.__x
                                 , self.__y
                                 , self.__z
                                 , self.__vx
                                 , self.__vy
                                 , self.__vz])
        super(ThreeDPosVelObs, self).__init__(time, obsVector)
