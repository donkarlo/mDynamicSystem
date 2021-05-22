from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.TransitionMatrix import TransitionMatrix
from mDynamicSystem.state.estimation.process.GaussianNoiseDecorator import GaussianNoiseDecorator
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.linearAlgebra.Vector import Vector


class ProcessModel(GaussianNoiseDecorator):
    ''''''
    def __init__(self
                 , previousState: State
                 , currentControlInput: Vector
                 , gaussianNoisePdf:Gaussian
                 , stateTransitionMatrix:TransitionMatrix):
        '''
        :param stateTransitionMatrix:
        :param gaussianNoisePdf:Gaussian for it covariance, use the covariance of the superstate
        '''
        self.__stateTransitionMatrix = stateTransitionMatrix
        self.__gaussianNoisePdf = gaussianNoisePdf
        super().__init__(previousState, currentControlInput,self.getPreviousNoise())

    def getPredictedState(self) -> State:
        '''
        :return:State
        '''
        predictedState:State = State(self.__stateTransitionMatrix.getNextMostProbableStateByState(self.getPreviousState()).getRefVec() + self.getPreviousNoise())
        return predictedState