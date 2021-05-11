from mMath.linearalgebra import Vector
from mMath.linearalgebra.Matrix import Matrix
from mDynamicSystem.state import State
from mDynamicSystem.state.estimation import ProcessModel as CoreStateEquation


class StateEquation(CoreStateEquation):
    def __init__(self
                 , curStateMatrix: Matrix = None
                 , prevStateVector: State = None
                 , previousNoiseVector: Vector = None
                 ):
        ''' x_k=Fx_{k-1}+E_{k-1}

        Parameters
        ----------
        __f:Matrix
            is F
        __b:Matrix
            is B
        '''
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
