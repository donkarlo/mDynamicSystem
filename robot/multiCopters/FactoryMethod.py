from mDynamicSystem.robot.multiCopters.HexaCopter import HexaCopter
from mDynamicSystem.robot.multiCopters.MultiCopter import MultiCopter
from mDynamicSystem.robot.multiCopters.QuadCopter import QuadCopter
from mDynamicSystem.robot.multiCopters.State import State


class FactoryMethod():
    def makeMultiCopter(self,rotorsNum:int)->MultiCopter:
        if rotorsNum == 4:
            return QuadCopter()
        elif rotorsNum == 6:
            return HexaCopter()
        else:
            raise Exception("I can't model a {}-Copter".format(rotorsNum))

########## CLIENT CODE#############
if __name__=="__main__":
    mcf = FactoryMethod()
    multiCopter = mcf.makeMultiCopter(4)

    # Polymorphism: We are coding to an interface we dont care if Hexa or Quad Copter is flying
    state = State(5,3,4,1,3,7)
    multiCopter.flyTo(state)

    print(type(multiCopter))

    # Asking for a model that doesnt exist
    try:
        nonExistantCopter = mcf.makeMultiCopter(5)
    except Exception as exp:
        print(exp.__str__())


