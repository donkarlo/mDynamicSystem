from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.State import State
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.data.probability.Pdf import Pdf
from mMath.data.probability.event.Vector import Vector
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace


class EuclideanMeasurementLikelihoodPdf(Pdf):
    def init(self, stateSpace:StateSpace, measurementModel:MeasurementModel):
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
            loopingMeasurement:Measurement = self.__measurementModel.getMeasurementRefVecWithoutNoise()
            distance = predictedMeasurementRefVec.getDistanceFrom(loopingMeasurement.getRefVec())
            sumDistances += distance
            if counter==0:
                minDistance = distance
            elif distance<minDistance:
                minDistance=distance
        return minDistance/sumDistances


