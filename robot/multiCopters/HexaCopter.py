from builtins import super

from mDynamicSystem.robot.multiCopters.MultiCopter import MultiCopter


class HexaCopter(MultiCopter):
    def __init__(self):
        super().__init__()
        print("A Hexa-Copter is added to your model")

    def getLoadCapacity(self) -> float:
        return 12

    def getStatePrecision(self) -> float:
        return 20
