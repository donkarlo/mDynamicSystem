from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian as GaussianPdf
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.TransitionMatrix import TransitionMatrix as StateTransitionMatrix


class GaussianNoiseStateTransitionMatrixModel(ProcessModel):
    def __init__(self,stateTransitionMatrix:StateTransitionMatrix,gaussianPdf:GaussianPdf):
        '''
        :param stateTransitionMatrix:
        :param gaussianPdf:
        '''
        self._stateTransitionMatrix:StateTransitionMatrix = stateTransitionMatrix
        self._gaussianPdf:GaussianPdf = gaussianPdf

    def getNextState(self) -> State:
        '''
        :return:
        '''
        return self._stateTransitionMatrix.getNextMostProbableStateByState(self._previousState)

    def getGaussianPdf(self)->GaussianPdf:
        ''''''
        return self._gaussianPdf