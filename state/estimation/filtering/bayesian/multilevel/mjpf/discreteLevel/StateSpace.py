from typing import List

from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreteLevel.State import State
from mMath.data.cluster.gng.Cluster import Cluster
from mMath.data.cluster.gng.examples.trajectory.Trajectory import Trajectory
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace


class StateSpace(StateSpace):
    def __init__(self):
        super().__init__()
        self.__clusters: List[Cluster] = self.__getClusters()
        loopingCluster: Cluster
        for loopingClusterKey in self.__clusters:
            superState: State = State(self.__clusters[loopingClusterKey].getVectors().getMeanVector(),
                                      self.__clusters[loopingClusterKey].getVectors().getVarianceVector())
            self.addState(superState)
    def __getClusters(self)->List[Cluster] :
        '''

        :return:
        '''
        trajectoryGng = Trajectory(3500)
        self.__clusters = trajectoryGng.getClusters()
        return self.__clusters
