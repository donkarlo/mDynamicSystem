from mDynamicSystem.robot.multiCopters.Flyable import Flyable
from mDynamicSystem.robot.multiCopters.State import State
from abc import abstractmethod

class MultiCopter(Flyable):
    '''To modle a Multi-Copter, a composit'''
    def __init__(self,state:State=None):
        self._state = state

    @abstractmethod
    def getLoadCapacity(self) -> float:
        pass

    @abstractmethod
    def getStatePrecision(self) -> float:
        pass

    def flyTo(self,state:State)->bool:
        print("I am flying to {},{},{},{},{},{}".format(state.x
                                                        ,state.y
                                                        ,state.z
                                                        ,state.vx
                                                        ,state.vy
                                                        ,state.vz))
        self._state = state

    def land(self) -> bool:
        pass

    def takeOff(self) -> bool:
        pass
