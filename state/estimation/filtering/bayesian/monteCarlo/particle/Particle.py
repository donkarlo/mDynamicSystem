import uuid

from mDynamicSystem.state.State import State


class Particle:
    def __init__(self
                 , state: float
                 , weight: float
                 , timeStep: int
                 ):
        self.__state:State = state
        self.__weight: float = weight
        self.__timeStep: int = timeStep
        self.__id:uuid.UUI= uuid.uuid1()

    def getState(self) -> State:
        ''''''
        return self.__state

    def getWeight(self)->float:
        return self.__weight

    def updateWeight(self,newWeight)->None:
        self.__weight = newWeight

    def updateState(self,state:State)->None:
        self.__state=State

    def update(self,state:State,weight:float):
        self.updateState(state)
        self.updateWeight(weight)