from mDynamicSystem.state.State import State
from mDynamicSystem.state.observation.Observation import Observation


class Particle:
    def __init__(self
                 , timeStep: int
                 , predictedState: State
                 , weight: float
                 , predictedObservation: Observation
                 , measuredObservation: Observation):
        self.__timeStep: int = timeStep
        # State-Observation pair: The information pair each two particle carry
        self.__predictedState: State = predictedState
        self.__weight: float = weight
        self.__predictedObservation: Observation = predictedObservation
        # Measured observation in current time
        self.__measuredObservation: Observation = measuredObservation

    def getState(self) -> State:
        return self.__predictedState

    def getMeasuredObservation(self) -> Observation:
        return self.__measuredObservation

    def getPredictedObservation(self) -> Observation:
        return self.__measuredObservation

    def getPredictedState(self) -> State:
        return self.__predictedState

    def getWeight(self):
        return self.__weight
