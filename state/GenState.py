from typing import List

from mMath.linearAlgebra.Vector import Vector
from mDynamicSystem.state import State


class Gs(State):
    '''Generalised states
    '''

    def __init__(self, derivativesVec:List[Vector]):
        self.__derivativesVec = derivativesVec
        pass
