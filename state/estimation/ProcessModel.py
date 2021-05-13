import abc
from mMath.linearalgebra import Vector
from mDynamicSystem.state import State


class ProcessModel(metaclass=abc.ABCMeta):
    '''Process Model: represents  x_k=f_k(x_{k_1},u_k,v_{k-1}) where f is an object of this class'''
    def __init__(self
         , previousState: State
         , controlInput: Vector
         , previousNoise: Vector
         , timeStep:int =None):
        '''

        :param previousState:
        :param controlInput:
        :param previousNoise:Vector it is independent and identically distributed
        :param timeStep: Note really important since we tag data in this class withh current and previous. The numer of measurement in measurement order, thats whu it should be integer. For example if between measurement 2 and three 2- secs wasted, between 4 and five can take 50 seconds and still timeInstance is 5.
        '''
        self.__previousState: State = previousState
        self.__controlInput:Vector = controlInput
        self.__previousNoise:Vector = previousNoise
        self.__timeStep: int = timeStep

    @abc.abstractmethod
    def getCurrentState(self) -> State:
        pass

    def getPreviousState(self)->State:
        return self.__previousState
