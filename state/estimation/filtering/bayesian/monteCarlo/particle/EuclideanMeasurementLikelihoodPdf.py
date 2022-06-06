from typing import List

from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.state.State import State
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.data.probability.Pdf import Pdf
from mMath.data.probability.event.Vector import Vector
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
from mMath.region.Region import Region


class EuclideanMeasurementLikelihoodPdf(Pdf):
    def __init__(self, stateSpace:StateSpace, measurementModel:MeasurementModel):
        '''

        :param stateSpace:
        :param measurementModel:
        :return:
        '''
        self.__stateSpace:StateSpace = stateSpace
        self.__measurementModel:MeasurementModel = measurementModel
    def getValueAt(self,predictedMeasurementRefVec: Vector)->float:
        '''
        :param state:
        :return:
        '''
        state:State
        minDistance:float
        sumDistances:float = 0
        for counter,loopingState in enumerate(self.__stateSpace.getStateSpaceSet()):
            self._measurementModel.updateState(loopingState)
            loopingMeasurement:Obs = self.__measurementModel.getMeasurementRefVecWithoutNoise()
            distance = predictedMeasurementRefVec.getDistanceFrom(loopingMeasurement.getRefVec())
            sumDistances += distance
            if counter==0:
                minDistance = distance
            elif distance<minDistance:
                minDistance=distance
        return minDistance/sumDistances

    def getSamples(self, samplesSize: int, intrestedRegion: Region) -> List:
        return None

    def getMean(self) -> float:
        return None

    def getCovarianceMatrix(self) -> float:
        return None

