from mMath.linearAlgebra.Vector import Vector


class Measurement:
    """Observation, many observations from different sensors may result in fewer states
    """

    def __init__(self, refVec: Vector, time: float = None):
        self.__refVec = refVec
        self.__time = time
