from mMath.linearAlgebra import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix
from mDynamicSystem.state import State
from mDynamicSystem.state.estimation.processModel import ProcessModel as MainStateModel


class StateModel(MainStateModel):
    def __init__(self
                 , curStateMatrix: Matrix = None
                 , prevStateVector: State = None
                 , previousNoiseVector: Vector = None
                 ):
        ''''''
        self.__currentStateMatrix:Matrix = curStateMatrix

    def getCurrentState(self
                        , prevStateVector: State = None
                        , prevInputVector: Vector = None
                        , prevProcessNoiseMatrix: Vector = None) -> State:
        """Get current state based on previous one
        """
        self.updatePreviousState(prevStateVector, prevInputVector, prevProcessNoiseMatrix)
        fPrevState = self.__currentStateMatrix * self.__prevStateVector
        bPrevInput = self.__controlMatrix * self.__prevNoise
        curState = fPrevState + bPrevInput + self.__prevW
        return curState

    def getProcessMatrix(self) -> Matrix:
        return self.__currentStateMatrix

    def getProcessNoiseCov(self) -> Matrix:
        return self.__processNoiseCovVector

    def getProcessMatrix(self):
        '''Train it if does not exist'''
        pass
