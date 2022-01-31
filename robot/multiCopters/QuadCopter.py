from builtins import super

from mDynamicSystem.robot.multiCopters.MultiCopter import MultiCopter


class QuadCopter(MultiCopter):
    def __init__(self):
        super().__init__()
        print("A quad-Copter is added to your model")

    def getLoadCapacity(self) -> float:
        return 8

    def getStatePrecision(self) -> float:
        return 16
