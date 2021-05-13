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

    def getState(self) -> State:
        ''''''
        return self.__predictedState

    def getWeight(self)->float:
        return self.__weight

    def updateWeight(self,newWeight)->None:
        self.__weight = newWeight