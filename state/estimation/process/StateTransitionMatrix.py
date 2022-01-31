from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.process.Model import Model
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix import TransitionMatrix as StochasticProcessStateTransitionMatrix


class StateTransitionMatrix(Model):
    ''''''
    def __init__(self,stateTransitionMatrix:StochasticProcessStateTransitionMatrix):
        '''

        :param model:
        :param stateTransitionMatrix:
        '''
        super().__init__()
        self._stateTransitionMatrix:StochasticProcessStateTransitionMatrix = stateTransitionMatrix

    def _getNextState(self)->State:
        '''
        :return:State
        '''
        nextStateId = self._stateTransitionMatrix.getNextMostProbableStateIdByStateId(self.getPreviousState().getId())
        stateSet:StateSpace = self._stateTransitionMatrix.getStateSpace()
        nextState:State = stateSet.getStateByStateId(nextStateId)
        return nextState.getRefVec()



