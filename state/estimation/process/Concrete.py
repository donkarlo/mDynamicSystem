from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mMath.probability import Pdf
from mMath.linearAlgebra.Vector import Vector


class Concrete(ProcessModel):
    def __init__(self
                 , previousState: State = None
                 , currentControlInput: Vector = None
                 , previousNoisePdf: Pdf = None
                 , timeStep:int = None):
        super().__init__(previousState
                         ,currentControlInput
                         ,previousNoisePdf
                         ,timeStep)


    def _getNextState(self)->Vector:
        return None