import abc
from mMath.linearAlgebra.Vector import Vector
from mDynamicSystem.state.State import State


class Model(metaclass=abc.ABCMeta):
    '''Process Model: represents  x_k=f_k(x_{k_1},u_k,v_{k-1}) where f is an object of this class'''
    def __init__(self
                 , previousState: State
                 , currentControlInput: Vector
                 , previousNoise: Vector
                 , timeStep:int =None):
        '''

        :param previousState:
        :param currentControlInput:
        :param previousNoise:Vector it is independent and identically distributed
        :param timeStep: Note really important since we tag data in this class withh current and previous. The numer of measurement in measurement order, thats whu it should be integer. For example if between measurement 2 and three 2- secs wasted, between 4 and five can take 50 seconds and still timeInstance is 5.
        '''
        self._previousState: State = previousState
        self._currentControlInput:Vector = currentControlInput
        self._previousNoise:Vector = previousNoise
        self._timeStep: int = timeStep

    @abc.abstractmethod
    def getNextState(self) -> State:
        pass

    def getPreviousState(self)->State:
        return self._previousState

    def getPreviousNoise(self):
        return self._previousNoise

