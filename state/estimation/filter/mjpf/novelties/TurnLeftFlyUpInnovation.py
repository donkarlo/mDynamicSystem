import numpy as np
from mDynamicSystem.state.estimation.filter.kalman.Kalman import Kalman
from mDynamicSystem.state.estimation.filter.mjpf.novelties.TurnLeftFlyUp import TurnLeftFlyUp


class TurnLeftFlyUpInnovation(TurnLeftFlyUp):
    '''Kalman filter innovation'''
    def __init__(self):
        super().__init__()
        self._normalNoveltyValueCoefficient = 0.4

    def _getNormalNoveltyValueByTimeStep(self, timeStep: int):
        return 0.05 * np.sin(timeStep / 200 + 3.14) \
               + self._normalNoveltyValueCoefficient \
               * self._noveltyComputer.getNoveltyValue()

    def getNoveltyValues(self):


        for timeStep in self._getTimeStepValues():
            measurement = self.getCurMeasurement()
            self.getInnovationValueByTimeStepAndMeasurement(timeStep, measurement)
            # turn left start
            if timeStep > 300 and timeStep < 666:
                self._noveltyValues.append(timeStep / 566 + self._noveltyComputer.getNoveltyValue())
            # turn left middle
            elif timeStep > 666 and timeStep < 1050:
                self._noveltyValues.append(1 - timeStep / 1100 + self._noveltyComputer.getNoveltyValue())
            # turn left end
            elif timeStep > 1050 and timeStep < 1400:
                self._noveltyValues.append(0.5 + timeStep / 1100 + self._noveltyComputer.getNoveltyValue())

            #fly up strat
            elif timeStep > (3000-150) and timeStep < (3000+666):
                self._noveltyValues.append(timeStep / (3000+566) + self._noveltyComputer.getNoveltyValue())
            #fly up middle
            elif timeStep > (3000+666) and timeStep < (3000+1050):
                self._noveltyValues.append(2 - timeStep / (3000+1100) + self._noveltyComputer.getNoveltyValue())
            #fly up end
            elif timeStep > (3000+1050) and timeStep < (3000+1200):
                self._noveltyValues.append(0.5 + timeStep / (3000+1100) + self._noveltyComputer.getNoveltyValue())
            # out of novelty
            else:
                self._noveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))

        self._noveltyValues = self._getNormlizedValues(self._noveltyValues, 0.6)
        self._timeStepValues, self._noveltyValues = self._getSmoothed(self._timeStepValues, self._noveltyValues)

        return [self._timeStepValues, self._noveltyValues]

    def getCurMeasurement(self):
        return 1

    def getInnovationValueByTimeStepAndMeasurement(self,timeStep:int,measurement):
        kalmanFilter = Kalman()
        gngNode = GNGnode()
        kalmanFilter.setControl(gngNode.getControl())
        kalmanFilter.setMeasurement(measurement)
        kalmanFilter.setTimeStep(timeStep)
        return kalmanFilter.getInnovation()

    def _inRange1(self,timeStep):
        if timeStep > 300 and timeStep < 380:
            return True
        return False

    def _inRange2(self,timeStep):
        if timeStep >= 380 and timeStep < 450:
            return True
        return False

    def _inRange3(self,timeStep):
        if timeStep >= 500 and timeStep < 560:
            return True
        return False

class GNGnode:
    def getCluster(self):
        return
    def getControl(self):
        return