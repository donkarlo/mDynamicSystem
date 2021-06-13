from mMath.data.probability.Pdf import Pdf
from mMath.linearAlgebra import Vector
from mDynamicSystem.state import State
from mDynamicSystem.measurement import Measurement
import abc

class Model(metaclass=abc.ABCMeta):
    '''z_k=h_k(x_k,u_k,n_k)
    - It's duty is to connect a given state to a measurement
    '''

    def __init__(self
                 , state: State = None
                 , noisePdf: Pdf = None
                 , controlInput: Vector = None
                 , timeStep:int=None):
        '''
        :param state:
        :param controlInput
        :param noise:
        :param timeStep:
        '''
        self._state: State = state
        self._noisePdf: Pdf = noisePdf
        self._controlInput: State = controlInput
        self._timeStep:int = timeStep


    @abc.abstractmethod
    def getMeasurementRefVecWithoutNoise(self) -> Vector:
        pass


    def getMeasurement(self) -> Measurement:
        ''''''
        return Measurement(self.getMeasurementRefVecWithoutNoise() + self.__getASampleNoise())



    def __getASampleNoise(self)->Vector:
        return self._noisePdf.getASample()

    def updateState(self,newState:State):
        '''

        :param newState:
        :return:
        '''
        self._state = newState

    def getState(self)->State:
        '''

        :return:
        '''
        return self._state

    def getNoisePdf(self)->Vector:
        '''

        :return:
        '''
        return self._noisePdf