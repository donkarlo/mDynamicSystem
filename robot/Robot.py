from typing import List

from mDynamicSystem.robot.actuator.Actuator import Actuator
from mDynamicSystem.robot.sensor import Sensor
from mDynamicSystem.state.estimation.process.Model import Model


class robot:
    ''''''
    def __init__(self, sensors:List[Sensor], actuator:List[Actuator], processModel:Model):
        '''

        :param sensors:
        :param actuator
        :param processModel:
        '''
        self.__sensors = sensors
        self.__processModel = processModel
        self.__actuator = actuator

    def updateProcessModel(self, processModel:Model):
        '''
        :param processModel:
        :return:
        '''
        self.__processModel = processModel