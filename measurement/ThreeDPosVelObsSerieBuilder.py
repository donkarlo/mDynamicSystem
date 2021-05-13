from mDynamicSystem.state import ObservationSerie
from mDynamicSystem.measurement import ThreeDPosVelObs


class ThreeDPosVelObsSerieBuilder():
    '''Just a builder for ObsSerie, it just appends!!'''

    def __init__(self):
        ''''''
        self.__obsList = []

    def append(self, threeDPosVel: ThreeDPosVelObs) -> None:
        ''''''
        self.__obsList.append(threeDPosVel)
        print(self.__obsList)

    def getObservationSerie(self) -> ObservationSerie:
        ''''''
        self.__observationSerie = ObservationSerie(self.__obsList)
        return self.__observationSerie
