from typing import List

from mMath.linearalgebra.Matrix import Matrix
from mMath.linearalgebra.Vector import Vector


class TransitionMatrix:
    ''''''
    def __init__(self,
                 inputData:Matrix
                 ,clusters:List[List[Vector]]):
        '''

        :param inputData:
        :param clusters: An array of points represnet in vectors
        '''
        self.__inputDate:Matrix = inputData
        self.__transitionMatrix:Matrix = None
        self.__clusters:List[List[Vector]] = clusters

    def getMatrix(self)->Matrix:
        ''''''
        if self.__transitionMatrix is None:
            for npRowInputData in self.__inputDate.getNpRows():
                pass

        return self.__transitionMatrix
