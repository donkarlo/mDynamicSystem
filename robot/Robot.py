from typing import List

from mDynamicSystem.robot.actuator.Actuator import Actuator
from mDynamicSystem.sensor import Sensor
from mDynamicSystem.state.estimation.ProcessModel import ProcessModel


class robot:
    ''''''
    def __init__(self,sensors:List[Sensor],actuator:List[Actuator],processModel:ProcessModel):
        '''

        :param sensors:
        :param actuator
        :param processModel:
        '''
        self.__sensors = sensors
        self.__processModel = processModel
        self.__actuator = actuator

    def updateProcessModel(self,processModel:ProcessModel):
        '''
        :param processModel:
        :return:
        '''
        self.__processModel = processModel