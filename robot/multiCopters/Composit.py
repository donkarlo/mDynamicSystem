from mDynamicSystem.robot.multiCopters.MultiCopter import MultiCopter
from mDynamicSystem.robot.multiCopters.FactoryMethod import FactoryMethod


class Composit(MultiCopter):
    def __init__(self):
        self._members = set()

    def add(self,member:MultiCopter)->None:
        self._members.add(member)

    def remove(self, member:MultiCopter):
        self._members.discard(member)

    def getLoadCapacity(self)->float:
        loadCapacity=0
        for member in self._members:
            loadCapacity+=member.getLoadCapacity()
        return loadCapacity

    def getStatePrecision(self) -> float:
        return 0

####### HERE STARTS THE CLIENT CODE###########
if __name__=="__main__":
    mcf = FactoryMethod()
    multiCopter2 = mcf.makeMultiCopter(4)
    multiCopter1 = mcf.makeMultiCopter(6)
    multiCopter3 = mcf.makeMultiCopter(6)
    multiCopterFleet = Composit()
    multiCopterFleet.add(multiCopter1)
    multiCopterFleet.add(multiCopter2)
    multiCopterFleet.add(multiCopter3)
    print("Now your load capacity of the whole fleet is: {}".format(multiCopterFleet.getLoadCapacity()))

    multiCopterFleet.remove(multiCopter2)
    print("Now your load capacity of the whole fleet is: {}".format(multiCopterFleet.getLoadCapacity()))

