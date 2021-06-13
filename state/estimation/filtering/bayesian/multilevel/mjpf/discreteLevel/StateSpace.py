from typing import List

from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreteLevel.State import State
from mMath.data.cluster.gng.Cluster import Cluster
from mMath.data.cluster.gng.examples.trajectory.Trajectory import Trajectory
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace


class StateSpace:
    def __init__(self):
        self._clusters: List[Cluster] = []
    def _getClusters(self)->List[Cluster] :
        '''

        :return:
        '''
        if len(self._stateSpace) == 0:
            trajectoryGng = Trajectory(200)
            self._clusters = trajectoryGng.getClusters()
        return self._clusters


    def getStateSpace(self)->StateSpace:
        '''
        Makes the each cluster mean and variance a state
        :return:
        '''
        loopingCluster:Cluster
        if self._stateSpace.getLength() == 0:
            for loopingCluster in self._getClusters():
                superState:State = State(loopingCluster.getVectors().getMeanVector(), loopingCluster.getVectors().getVarianceVector())
                self._stateSpace.addState(superState)
        return self._stateSpace