from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.process.Model import Model
from mDynamicSystem.state.estimation.process.decorator.Decorator import Decorator
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix import TransitionMatrix as StochasticProcessStateTransitionMatrix


class StateTransitionMatrix(Decorator):
    ''''''
    def __init__(self, model:Model, stateTransitionMatrix:StochasticProcessStateTransitionMatrix):
        '''

        :param model:
        :param stateTransitionMatrix:
        '''
        super().__init__(model)
        self._model._stateTransitionMatrix:StateTransitionMatrix = stateTransitionMatrix

    def _getPredictedStateRefVecWithoutNoise(self)->State:
        '''
        :return:State
        '''
        nextStateId = self._model._stateTransitionMatrix.getNextMostProbableStateIdByStateId(self._model.getPreviousState())
        stateSet:StateSpace = self._stateTransitionMatrix.getStateSpace()
        nextState:State = stateSet.getStateByStateId(nextStateId)
        return nextState.getRefVec()



