from mDynamicSystem.robot.multiCopter.MultiCopter import MultiCopter
from mDynamicSystem.robot.multiCopter.FactoryMethod import FactoryMethod


class Decorator(MultiCopter):
    def __init__(self,multiCopter:MultiCopter):
        self._multiCopter:MultiCopter = multiCopter

####### HERE STARTS THE CLIENT CODE###########
if __name__=="__main__":
    mcf = FactoryMethod()
    multiCopter = mcf.makeMultiCopter(4)

    from mDynamicSystem.robot.multiCopter.LidarDecorator import LidarDecorator
    lidarDecoratedCopter = LidarDecorator(multiCopter)

    from mDynamicSystem.robot.multiCopter.GpsDecorator import GpsDecorator
    gpsLidarDecoratedCopter = GpsDecorator(lidarDecoratedCopter)

    print("The precision is: {}".format(gpsLidarDecoratedCopter.getStatePrecision()))

