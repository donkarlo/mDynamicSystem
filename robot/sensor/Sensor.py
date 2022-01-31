import abc

from mDynamicSystem.state.State import State
from mDynamicSystem.state.measurement.Model import Model
from mDynamicSystem.obs.Obs import Obs


class Sensor:
    def __init__(self
                 , measurementModel:Model):
        '''

        :param measurementModel:
        '''
        self.__measurementModel = measurementModel

    @abc.abstractmethod
    def getMeasurementLikelihoodConditionOnState(self, measurement:Obs, state:State)->float:
        '''
        p(z_k|x_k)
        :param measurement:
        :param state:
        :return:
        '''
        pass