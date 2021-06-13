from typing import List

from mDynamicSystem.state.estimation.filtering.StateProbability import StateProbability


class StateSpaceProbabilities:
    def __init__(self,stateSpaceProbabilitiesList:List[StateProbability]):
        '''@todo check if the sum of all probabilities is 1
        '''
        self._stateSpaceProbabilitiesList:List[StateProbability] = stateSpaceProbabilitiesList

    def getStateSpaceProbabilitiesList(self)->List[StateProbability]:
        return self._stateSpaceProbabilitiesList