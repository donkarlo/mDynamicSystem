from mMath.linearalgebra import Vector
from mDynamicSystem.state import State
from mDynamicSystem.state.observation import Observation
import abc

class MeasurementEquation(abc.ABCMeta):
    '''z_k=Hx_{k}+v_{k}, v_{k}~N(0,R)

    '''

    def __init__(self
                 , currentState: State
                 , currentControlInput: Vector
                 , currentNoise: Vector
                 ,timeStep:int):
        '''
        :param currentState:
        :param currentControlInput
        :param currentNoise:
        :param timeStep:
        '''
        self.__currentState: State = currentState
        self.__currentControlInput: State = currentControlInput
        self.__currentNoise:Vector = currentNoise
        self.__timeStep: int = timeStep

    @abc.abstractmethod
    def getCurrentObservation(self) -> Observation:
        pass

