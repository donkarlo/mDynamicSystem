from typing import List

from mDynamicSystem.control.Input import Input


class InputsSerie():
    """A time serie of measurement vectors
    """

    def __init__(self, actionList:List[Input]):
        self.__inputList = actionList

    def getByIndex(self, index: int) -> int:
        return self.__inputList[index]

    def getActionsList(self):
        return self.__inputList
