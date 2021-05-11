from typing import List

from mDynamicSystem.state.State import State


class Serie:
    def __init__(self):
        self.__statesList:List[State]=None

    def addState(self,state:State):
        self.__statesList.append(state)

    def getStates(self)->List[State]:
        return self.__statesList

    def getLastState(self)->State:
        ''''''
        return self.__statesList.getByIdx(self.getLastStateIndex())

    def getLastStateIndex(self):
        return len(self.__statesList)-1