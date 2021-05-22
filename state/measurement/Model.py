from mMath.linearAlgebra import Vector
from mDynamicSystem.state import State
from mDynamicSystem.measurement import Measurement
import abc

class Model(abc.ABCMeta):
    '''z_k=h_k(x_k,u_k,n_k)
    '''

    def __init__(self
                 , state: State
                 , controlInput: Vector
                 , noise: Vector
                 , timeStep:int=None):
        '''
        :param state:
        :param controlInput
        :param noise:
        :param timeStep:
        '''
        self._state: State = state
        self._controlInput: State = controlInput
        self._noise:Vector = noise
        self._timeStep:int = timeStep

        self._measurement:Measurement = None

    @abc.abstractmethod
    def getMeasurement(self) -> Measurement:
        pass

    @abc.abstractmethod
    def getMeasurementWithoutNoise(self)->Measurement:
        pass

    def updateState(self,newState:State):
        self._state = newState

    def getState(self)->State:
        return self._state

    def getNoise(self)->Vector:
        return self._noise