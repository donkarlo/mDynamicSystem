import abc

from mMath.data.probability.Pdf import Pdf
from mMath.linearAlgebra.Vector import Vector
from mDynamicSystem.state.State import State


class Model(metaclass=abc.ABCMeta):
    '''Process Model: represents  x_k=f_k(x_{k_1},u_k,v_{k-1}) where f is an object of this class'''
    def __init__(self
                 , previousState: State = None
                 , currentControlInput: Vector = None
                 , previousNoisePdf:Pdf = None
                 , timeStep:int =None):
        '''

        :param previousState:
        :param currentControlInput:
        :param previousNoise:Vector/Pdf it is independent and identically distributed
        :param timeStep: Note really important since we tag data in this class withh current and previous. The numer of measurement in measurement order, thats whu it should be integer. For example if between measurement 2 and three 2- secs wasted, between 4 and five can take 50 seconds and still timeInstance is 5.
        '''
        self._previousState: State = previousState
        self._currentControlInput:Vector = currentControlInput
        self._previousNoisePdf = previousNoisePdf
        self._timeStep: int = timeStep

        self._previousNoiseSample = None
        self._nextStateWithoutNoise = None

    @abc.abstractmethod
    def getNextStateWithoutNoise(self)->State:
        '''
        When we say nextState, it means that control is already encoded inside
        :return:
        '''
        pass

    def update(self
               , previousState: State
                 , currentControlInput: Vector
                 , previousNoisePdf:Pdf
                 , timeStep:int =None):
        self._previousState: State = previousState
        self._currentControlInput: Vector = currentControlInput
        self._previousNoisePdf = previousNoisePdf
        self._timeStep: int = timeStep

    def getCurrentControlInput(self)->Vector:
        return self._currentControlInput

    def getNextMostProbableState(self) -> State:
        return self.getNextStateWithoutNoise()+self.getPreviousNoise()

    def getPreviousState(self)->State:
        '''
        :return:
        '''
        return self._previousState


    def getPreviousNoise(self)->Vector:
        '''
        :return:
        '''
        self._previousNoiseSample = self._previousNoisePdf.getSamples()[0]
        return self._previousNoiseSample

    def getPreviousNoisePdf(self)->Pdf:
        return self._previousNoisePdf

    def getTimeStep(self)->int:
        return self._timeStep

