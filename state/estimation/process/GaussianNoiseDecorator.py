from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.process.Model import Model
from mMath.linearAlgebra.Vector import Vector
import numpy as np

class GaussianNoiseDecorator(Model):
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