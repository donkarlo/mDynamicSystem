from typing import List

from mDynamicSystem.control import Input
from mDynamicSystem.control.InputsSerie import InputsSerie


class InputsSerieBuilder():
    def __init__(self):
        self.__actionList:List[Input]=[]

    def appendAction(self, action: Input) -> None:
        self.__actionList.appendAction(action)

    def getActionSerie(self) -> InputsSerie:
        self.__actionSerie = InputsSerie(self.__actionList)
        return self.__actionSerie
