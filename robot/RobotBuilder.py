from mDynamicSystem.robot.actuator.Actuator import Actuator
from mDynamicSystem.robot.Robot import Sensor


class RobotBuilder:
    def __init__(self):
        ''''''
        self.__sensors = []
        self.__actuators = []

    def addSensor(self,sensor:Sensor):
        '''
        :param sensor:
        :return:
        '''
        self.__sensors.append(sensor)


    def addActuator(self,actuator:Actuator):
        '''
        :param actuator:
        :return:
        '''
        self.__actuators.append(actuator)
