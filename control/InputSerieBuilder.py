from typing import List

from mDynamicSystem.control import Input
from mDynamicSystem.control.InputSerie import ActionSerie


class ActionSerieBuilder():
    def __init__(self):
        self.__actionList:List[Input]=[]

    def appendAction(self, action: Input) -> None:
        self.__actionList.appendAction(action)

    def getActionSerie(self) -> ActionSerie:
        self.__actionSerie = ActionSerie(self.__actionList)
        return self.__actionSerie
