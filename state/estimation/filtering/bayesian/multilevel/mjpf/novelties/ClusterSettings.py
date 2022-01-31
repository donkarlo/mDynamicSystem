class ClusterSettings:
    def __init__(self,numOfClusters:int,noveltyValueCoefficient:float):
        self._numOfClusters = numOfClusters
        self._noveltyValueCoeffcient = noveltyValueCoefficient

    def getNoveltyValueCoefficeint(self)->float:
        return self._noveltyValueCoeffcient

