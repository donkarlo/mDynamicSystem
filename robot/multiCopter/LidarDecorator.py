from mDynamicSystem.robot.multiCopter.Decorator import Decorator


class LidarDecorator(Decorator):
    def getStatePrecision(self) -> float:
        return self._multiCopter.getStatePrecision()+10

    def getLoadCapacity(self) -> float:
        return 0