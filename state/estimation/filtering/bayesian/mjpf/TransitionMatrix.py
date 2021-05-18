from typing import List

from mDynamicSystem.state.State import State
from mMath.data.cluster.Cluster import Cluster
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.FromStateSerieBuilder import FromStateSerieBuilder
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.TransitionMatrix import TransitionMatrix as MainTransitionMatrix
from mMath.linearAlgebra.matrix.Matrix import Matrix




class TransitionMatrix(MainTransitionMatrix):
    ''''''
    def __init__(self
                 , clusters:List[Cluster]):
        '''

        :param measurementDataMatrix:
        :param clusters: An array of points represnet in vectors
        '''
        self.__transitionMatrix:Matrix = None
        self.__clusters:List[Cluster] = clusters
        #state sequence

        self.__stateSerie:StateSerie = None
        trMatBld:FromStateSerieBuilder=FromStateSerieBuilder()
        super().__init__(self.__stateSerie)

    def getMatrix(self)->Matrix:
        ''''''
        if self.__transitionMatrix is None:
            for npRowInputData in self.__inputDate.getNpRows():
                pass

        return self.__transitionMatrix


