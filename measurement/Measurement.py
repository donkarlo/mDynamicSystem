from mMath.linearAlgebra.Vector import Vector


class Measurement:
    """Observation, many observations from different sensors may result in fewer states
    """

    def __init__(self, time: float, obsVector: Vector):
        self.__time = time
        self.__obsVector = obsVector

    def updateByIndex(self, idx: int, updateVal: float):
        self.__obsVector[idx] = updateVal

    def getComponentByIdx(self, idx):
        return self.__obsVector[idx]

    def getTime(self) -> float:
        return self.__time
