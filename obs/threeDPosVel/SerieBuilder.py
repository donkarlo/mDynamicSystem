from typing import List

from mDynamicSystem.obs.threeDPosVel import Obs as ThreeDPosVel
from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.obs.Serie import Serie as obsSerie


class SerieBuilder():
    '''Just a builder for ObsSerie, it just appends!!'''

    def __init__(self):
        ''''''
        self.__obsList:List[Obs] = []
        self.__obsSerie:obsSerie = None

    def append(self, threeDPosVel: ThreeDPosVel) -> None:
        ''''''
        self.__obsList.append(threeDPosVel)
        print(self.__obsList)

    def getObservationSerie(self) -> obsSerie:
        ''''''
        self.__obsSerie = obsSerie(self.__obsList)
        return self.__obsSerie
