from IPython.core.display import Math

from mMath.linearAlgebra.Vector import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix


class MahanLobis:
    def __init__(self,mean:Vector,covarianceMatrix:Matrix,point:Vector):
        self._mean = mean
        self._covarianceMatrix = covarianceMatrix
        self._point = point

    def getValue(self):
        minus = (self._point-self._mean)
        return Math.sqrt(minus*self._covarianceMatrix^-1*minus)