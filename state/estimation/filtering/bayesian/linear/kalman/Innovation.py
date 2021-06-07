from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state import State
from mMath.linearAlgebra.Vector import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix


class Innovation:
    '''y^{~}_{k} = z_k-Hx^{^-}_{k}

    Parameters
    ----------

    Returns
    _______

    '''

    def __init__(self
                 , processMatrix: Matrix
                 , currentObservation: Measurement
                 , priorCurrentEstimatedState: State):
        self._processMatrix = processMatrix
        self.__currentObservation = currentObservation
        self.__priorCurrentEstimatedState = priorCurrentEstimatedState

    def getInnovation(self) -> Vector:
        yK = self.__currentObservation - self.__processMatrix * self._priorCurrentEstimatedState
        return yK
