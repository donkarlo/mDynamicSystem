import string
from typing import List

from mUtility.database.file.RowsStartWithColNames import RowsStartWithColNames
from mDynamicSystem.obs.Serie import Serie as ObsSerie
from mDynamicSystem.obs.threeDPosVel.Obs import Obs as ThreeDPosVelObs
from mDynamicSystem.obs.threeDPosVel.SerieBuilder import SerieBuilder


class PosToVelObsSerieBuilderFromRowsStartWithColNames:
    '''This class is responsible for Building and obs serie from a topic dumped text file and save it
    '''

    def __init__(self
                 , dumpedTextFile: RowsStartWithColNames
                 , posColNames: List[str]
                 , linelimit: int):
        '''
        Parameters
        ----------
        posColNames: array
            XYZ must be in order
        '''
        self.__rowStartWithColNamesFile:RowsStartWithColNames = dumpedTextFile
        self.__posColNames:List = posColNames
        self.__obsSerie:ObsSerie = None
        self.__linelimit:int = linelimit

    def getObsSerie(self) -> ObsSerie:
        '''

        '''
        if self.__obsSerie == None:
            fileToReadFrom = open(self.__rowStartWithColNamesFile.getFilePath(), "r")
            threeDPosVelObsSerieBuilder = SerieBuilder()
            for lineCounter, curLine in enumerate(fileToReadFrom):
                if lineCounter > self.__linelimit:
                    break
                elif lineCounter == 0:
                    continue
                elif lineCounter == 1:
                    curTime, curXObs, curYObs, curZObs = self.getTimeXyzObsFromStrLine(curLine)
                    curThreeDPosVelObs = ThreeDPosVelObs(curTime, curXObs, curYObs, curZObs, 0, 0, 0)
                    threeDPosVelObsSerieBuilder.append(curThreeDPosVelObs)  # we are fixing index 0
                    continue
                prvObs = threeDPosVelObsSerieBuilder.getObservationSerie().getMeasurementByIndex(lineCounter - 2)
                prvTime, prvXObs, prvYObs, prvZObs = float(prvObs.getComponentByIndex(0)), float(
                    prvObs.getComponentByIndex(1)), float(prvObs.getComponentByIndex(2)), float(prvObs.getComponentByIndex(3))
                curTime, curXObs, curYObs, curZObs = self.getTimeXyzObsFromStrLine(curLine)

                # timeDiff = (curTime - prvTime) / 1000000000000
                timeDiff = curTime - prvTime
                curXVelObs, curYVelObs, curZVelObs = [(curXObs - prvXObs) / timeDiff
                    , (curYObs - prvYObs) / timeDiff
                    , (curZObs - prvZObs) / timeDiff]
                curThreeDPosVelObs = ThreeDPosVelObs(curTime
                                         , curXObs
                                         , curYObs
                                         , curZObs
                                         , curXVelObs
                                         , curYVelObs
                                         , curZVelObs)
                threeDPosVelObsSerieBuilder.append(curThreeDPosVelObs)
                # update the first row with the second row velocity
                if lineCounter == 2:
                    firstThreeDPosVelObs = threeDPosVelObsSerieBuilder.getObservationSerie().getMeasurementByIndex(0)
                    firstThreeDPosVelObs.updateComponentByIndex(3, curXVelObs)
                    firstThreeDPosVelObs.updateComponentByIndex(4, curYVelObs)
                    firstThreeDPosVelObs.updateComponentByIndex(5, curZVelObs)
            self.__obsSerie = threeDPosVelObsSerieBuilder.getObservationSerie()
        return self.__obsSerie

    def saveToFileWithTime(self, filePath: str, sep=",")->None:
        ''''''
        # save to the file from here
        with open(filePath, 'w') as f:
            obs: ThreeDPosVelObs
            for obs in self.getObsSerie().getMeasurementList():
                f.write("{}{}{}{}{}{}{}{}{}{}{}{}{}\n".format(
                    obs.getTime(), sep,
                    obs.getComponentByIndex(0), sep,
                    obs.getComponentByIndex(1), sep,
                    obs.getComponentByIndex(2), sep,
                    obs.getComponentByIndex(3), sep,
                    obs.getComponentByIndex(4), sep,
                    obs.getComponentByIndex(5)))

    def saveToFileWithoutTime(self, filePath: str, sep=",")->None:
        ''''''
        # save to the file from here
        with open(filePath, 'w') as f:
            obs: ThreeDPosVelObs
            for obs in self.getObsSerie().getMeasurementList():
                f.write("{}{}{}{}{}{}{}{}{}{}{}\n".format(
                    obs.getComponentByIndex(0), sep,
                    obs.getComponentByIndex(1), sep,
                    obs.getComponentByIndex(2), sep,
                    obs.getComponentByIndex(3), sep,
                    obs.getComponentByIndex(4), sep,
                    obs.getComponentByIndex(5)))

    def getTimeXyzObsIndexes(self) -> list:
        '''
        '''
        timeIndex = 0
        xColIndex = self.__rowStartWithColNamesFile.getColIndexByName(self.__posColNames[0])
        yColIndex = self.__rowStartWithColNamesFile.getColIndexByName(self.__posColNames[1])
        zColIndex = self.__rowStartWithColNamesFile.getColIndexByName(self.__posColNames[2])
        return [timeIndex, xColIndex, yColIndex, zColIndex]

    def getTimeXyzObsFromStrLine(self, line: str) -> list:
        '''
        '''
        curLineList = line.split(",")
        timeIndex, xIndex, yIndex, zIndex = self.getTimeXyzObsIndexes()
        time = float(curLineList[timeIndex])
        curXObs = float(curLineList[xIndex])
        curYObs = float(curLineList[yIndex])
        curZObs = float(curLineList[zIndex])

        return [time, curXObs, curYObs, curZObs]
