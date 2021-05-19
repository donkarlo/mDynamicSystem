from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.Decorator import Decorator
from mMath.data.probability.continous.Gaussian import Gaussian
import numpy as np

from mMath.linearAlgebra.Vector import Vector


class GaussianNoiseDecorator(Decorator):
    def __init__(self, gaussianPdf:Gaussian):
        '''
        :param gaussianPdf:
        '''
        self.__gaussianPdf = gaussianPdf
        self.setNoise()
        super().__init__(self._model)

    def setNoise(self)->None:
        '''
        :return:
        '''
        stateRefVecComponentsNum = np.shape(self._model.getState().getRefVec().getNpRows())[0]
        vector = Vector(np.random.multivariate_normal(self.__gaussianPdf.getMean()
                                                      , self.__gaussianPdf.getCovariance()
                                                      , stateRefVecComponentsNum).T)
        self._model._noise:Vector = vector
