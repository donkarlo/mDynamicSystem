from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.processModel.ProcessModel import ProcessModel
from mMath.linearAlgebra.Vector import Vector
import numpy as np

class ProcessModelWithGaussianNoise(ProcessModel):
    def __init(self
               , previousState: State
               , currentControlInput: Vector):
        '''

        :param previousState:
        :param currentControlInput:
        :return:
        '''
        previousNoise = self.getPreviousNoise()
        super().__init__(previousState
                         ,currentControlInput
                         ,previousNoise)

    def getPreviousNoise(self)->Vector:
        '''
        @todo Generate as many as the state from normal distribution
        :return:Vector
        '''
        stateRefVecComponentsNum = np.shape(self.getPreviousState().getRefVec().getNpRows())[0]
        vector = Vector(np.random.multivariate_normal(self.__gaussianNoisePdf.getMean()
                                              ,self.__gaussianNoisePdf.getCovariance()
                                              ,stateRefVecComponentsNum).T)
        return vector