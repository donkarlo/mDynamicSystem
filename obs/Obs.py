from mMath.linearAlgebra.Vector import Vector


class Obs(Vector):
    """Observation, many observations from different sensors may result in fewer states
    """

    def __init__(self,time: float,refVecList):
        self.__time = time
        super().__init__(refVecList)

    def getTime(self):
        return self.__time
