import abc

from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.MeasurementModel import MeasurementModel
from mDynamicSystem.measurement.Measurement import Measurement


class Sensor:
    def __init__(self
                 ,measurementModel:MeasurementModel):
        '''

        :param measurementModel:
        '''
        self.__measurementModel = measurementModel

    @abc.abstractmethod
    def getMeasurementLikelihoodConditionOnState(self,measurement:Measurement,state:State)->float:
        '''
        p(z_k|x_k)
        :param measurement:
        :param state:
        :return:
        '''
        pass