from mMath.linearAlgebra import Vector
from mDynamicSystem.state import State
from mDynamicSystem.measurement import Measurement
import abc

class MeasurementModel(abc.ABCMeta):
    '''z_k=h_k(x_k,u_k,n_k)
    '''

    def __init__(self
                 , currentState: State
                 , currentControlInput: Vector
                 , currentNoise: Vector
                 ,timeStep:int=None):
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
    def getCurrentObservation(self) -> Measurement:
        pass

