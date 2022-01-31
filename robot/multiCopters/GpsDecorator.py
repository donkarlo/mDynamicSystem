from mDynamicSystem.robot.multiCopters.Decorator import Decorator


class GpsDecorator(Decorator):
    def getStatePrecision(self) -> float:
        return self._multiCopter.getStatePrecision()+50

    def getLoadCapacity(self) -> float:
        return 0