from typing import List

from mDynamicSystem.control import Input


class ActionSerie():
    """A time serie of observation vectors
    """

    def __init__(self, actionList:List[Input]):
        self.__actionList = actionList

    def getByIdx(self, index: int) -> int:
        return self.__actionList[index]

    def getActionsList(self):
        return self.__actionList
