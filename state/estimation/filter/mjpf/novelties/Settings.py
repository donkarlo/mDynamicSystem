from mDynamicSystem.state.estimation.filter.mjpf.novelties.ClusterSettings import ClusterSettings


class Settings:
    def __init__(self):
        pass

    def setClusterSettings(self,clusterSettings:ClusterSettings):
        '''
        :param clusterSettings:
        :return:
        '''
        self._clusterSettings = clusterSettings

    def getClusterSettings(self)->ClusterSettings:
        '''
        :return:
        '''
        return self._clusterSettings