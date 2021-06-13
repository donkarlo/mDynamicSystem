from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.TrainingMeasurementSerie import \
    TrainingMeasurementSerie
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreteLevel.State import State
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mMath.data.timeSerie.stochasticProcess.state.SerieBuilder import SerieBuilder
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.FromStateSerieBuilder import FromStateSerieBuilder
from mDynamicSystem.state.estimation.process.decorator.StateTransitionMatrix import StateTransitionMatrix



class TransitionMatrix:
    def __init__(self):

        #for training the transition matrix
        self._trainingMeasurementSerie:TrainingMeasurementSerie = TrainingMeasurementSerie()
        self._stateSpace:StateSpace = None
        ''''''


    def _getMeasurementBelongingState(self, measurement:Measurement)->State:
        '''
        To which superstate this measurement belongs?
        :param measurement:
        :return:
        '''
        measurementRefVec = measurement.getRefVec()
        loopingSuperState: State
        superStateFoundForMeasurement = False
        belongingSuperState = None
        for loopingSuperState in self._getStateSpace():
            if loopingSuperState.getCenter().getDistanceFrom(measurementRefVec) < 3 * loopingSuperState.getStandardDeviation().getNorm():
                belongingSuperState = loopingSuperState
                superStateFoundForMeasurement = True
        if superStateFoundForMeasurement == False:
            belongingSuperState = self._getStateSpace().getARandomMember()
        return belongingSuperState

    def _getStateSerie(self)->StateSerie:
        '''
        :return:
        '''
        if self._stateSerie is None:
            stateSerieBuilder: SerieBuilder = SerieBuilder(self._getStateSpace())
            loopingOfflineMeasurement: Measurement
            for loopingOfflineMeasurement in self._trainingMeasurementSerie.getMeasurementList():
                stateSerieBuilder.appendState(self._getMeasurementBelongingState(loopingOfflineMeasurement))
            self._stateSerie = stateSerieBuilder.getSerie()
        return self._stateSerie

    def getTransitionMatrix(self)->StateTransitionMatrix:
        '''
        :return:
        '''
        transitionMatrix = FromStateSerieBuilder(self._getStateSerie())
        return transitionMatrix
