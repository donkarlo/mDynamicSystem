from mMath.data.probability.Pdf import Pdf
from mMath.linearAlgebra import Vector
from mDynamicSystem.state import State
from mDynamicSystem.measurement import Measurement
import abc

class Model(abc.ABCMeta):
    '''z_k=h_k(x_k,u_k,n_k)
    - It's duty is to connect a given state to a measurement
    '''

    def __init__(self
                 , state: State
                 , controlInput: Vector
                 , noisePdf: Pdf
                 , timeStep:int=None):
        '''
        :param state:
        :param controlInput
        :param noise:
        :param timeStep:
        '''
        self._state: State = state
        self._controlInput: State = controlInput
        self._noisePdf:Pdf = noisePdf
        self._timeStep:int = timeStep

        self._measurement:Measurement = None
        self._lastDarwnNoise = None

    @abc.abstractmethod
    def getMeasurementWithoutNoise(self) -> Measurement:
        pass


    def getMeasurement(self) -> Measurement:
        ''''''
        return Measurement(self.getMeasurementWithoutNoise().getRefVec()+self.getASampleNoise())



    def getASampleNoise(self)->float:
        self._lastDarwnNoise = self._noisePdf.getASample()
        return self._lastDarwnNoise

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

    def getNoise(self)->Vector:
        '''

        :return:
        '''
        return self._noise