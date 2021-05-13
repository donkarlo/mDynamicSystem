import abc
import numpy as np

from numpy import ndarray

from mDynamicSystem.state.estimation.ProcessModel import ProcessModel as MainProcessModel
from mDynamicSystem.state import State
from mMath.linearalgebra.Matrix import Matrix
from mMath.linearalgebra.Vector import Vector


class ProcessModel(MainProcessModel):

    def __init__(self
                 ,transitionMatrix:Matrix
                 , previousState: State
                 , controlInput: Vector
                 , previousNoise: Vector
                 , timeStep: int = None
                 ):
        '''

        :param transitionMatrix:
        :param previousState: State here is a cluster
        :param controlInput:
        :param previousNoise:
        :param timeStep:
        '''
        self.__transitionMatrix:Matrix = transitionMatrix
        super().__init__(previousState,controlInput,previousNoise,timeStep)

    def getCurrentState(self) -> State:
        ''''''
        npRow:np.ndarray = self.__transitionMatrix.getNpRowByIndex(self.__getTransitionMatrixRowIndexForPreviousState())
        return np.max(npRow)

    def __getTransitionMatrixRowIndexForPreviousState(self)->int:
        '''A mechanism for finding the row related to given state in transition matrix'''
        return None


