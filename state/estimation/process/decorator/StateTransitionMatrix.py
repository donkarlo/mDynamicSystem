from mDynamicSystem.state.estimation.process.Model import Model
from mDynamicSystem.state.estimation.process.decorator.Decorator import Decorator
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.data.timeSerie.stochasticProcess.state.StateSet import StateSet
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix import TransitionMatrix as StochasticProcessStateTransitionMatrix


class StateTransitionMatrix(Decorator):
    ''''''
    def __init__(self, model:Model, stateTransitionMatrix:StochasticProcessStateTransitionMatrix):
        super().__init__(model)
        self._model._stateTransitionMatrix:StateTransitionMatrix = stateTransitionMatrix

    def getNextStateWithoutNoise(self)->State:
        '''
        :return:State
        '''
        nextStateId = self._model._stateTransitionMatrix.getNextMostProbableStateIdByStateId(self._model.getPreviousState())
        stateSet:StateSet = self._stateTransitionMatrix.getStateSet()
        nextState = stateSet.getStateByStateId(nextStateId)
        return nextState



