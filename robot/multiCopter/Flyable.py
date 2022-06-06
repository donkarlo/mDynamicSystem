from abc import ABC, abstractmethod

from mDynamicSystem.robot.multiCopter.State import State


class Flyable(ABC):
    '''Interface for Flyable objects'''
    @abstractmethod
    def flyTo(self,state:State)->bool:
        pass

    @abstractmethod
    def land(self)->bool:
        pass

    @abstractmethod
    def takeOff(self)->bool:
        pass
