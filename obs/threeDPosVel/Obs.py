import numpy as np

from mDynamicSystem.obs.Obs import Obs


class Obs(Obs):
    def __init__(self, time: float
                 , x: float
                 , y: float
                 , z: float
                 , vx: float
                 , vy: float
                 , vz: float):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__vx = vx
        self.__vy = vy
        self.__vz = vz
        measurementVector = np.array([self.__x
                                 , self.__y
                                 , self.__z
                                 , self.__vx
                                 , self.__vy
                                 , self.__vz])
        super().__init__(time, measurementVector)
