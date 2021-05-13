import abc
from typing import List
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.data.probability.Event import Event
from mMath.data.probability.Pdf import Pdf
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.linearalgebra.Matrix import Matrix
from mMath.linearalgebra.Vector import Vector
from mDynamicSystem.state import State
from mDynamicSystem.state.Serie import Serie as StatesSerie
from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.measurement import MeasurementsSerie


class Filter(Estimation):
    ''' Byesian Filtering is a form of Estimation in which the intersection of measurement distribution and predicted, next state distribuation makes restricts the answer set
    - Byesian Filter refine estimates everytime a new measurement is achieved
    In each Byesian Filter our goal is to compute the posterior which is state estimation after measurement (self.getPosterior)
     first we predict () then we observe and update according to what is observed
    What is the PDF of state with a given set of measurement
    if the variables are normally distributed and the transitions are linear, the Bayes filtering becomes equal to the Kalman filtering.'''

    def __init__(self):
        # Estimations improve by receiving more measurement
        self.__observationSerie: MeasurementsSerie = None
        #History of states
        self._stateSerie: StatesSerie = None
        # p(x_k|z_{1:k-1})
        self.__prior: float = None
        # Acts as a normalizer, a constant which could be regarded as the probability of the measurement, sum(p(z_{1:k}))
        self.__marginalLikelihood: float = None

    def addObservation(self, observation: Measurement) -> None:
        '''

        :param observation: Observation
        :return:
        '''


        self.__observationSerie.appendObservation(observation)
        self._onAddMeasurement(observation)
        self._predict()
        self._update()

    @abc.abstractmethod
    def _onAddMeasurement(self, observation:Measurement):
        '''

        :param observation:
        :return:
        '''
        pass


    def _predict(self) -> State:
        '''First setp: predict prior to measurement using the StateEquation'''
        self._updatePrior()
        self._updateLikelihood()

    @abc.abstractmethod
    def _update(self, observation: Measurement) -> State:
        '''Second step: Update prediction, refining the _pridct sing bayes theorem'''
        pass

    def getCurrentPosterior(self):
        '''No stop condition is needed. the last measurement is the s'''
        currentPosterior = (self._getLikelihood() * self._getCurrentPrior()) / self._getMarginalLikelihood()
        return currentPosterior

    def _getCurrentPrior(self) -> float:
        ''''''
        pdf = Pdf()
        priorSum: float = 0
        stateCounter = 0
        for state in self._stateSerie:
            priorSum += pdf.getValueByEvent(
                Event(state).conditionedOn(Event(self._stateSerie.getByIndex(stateCounter - 1))))
            stateCounter += 1
        self.__prior = priorSum
        return priorSum

    def _getCurrentLikelihood(self) -> float:
        '''A knowledge which we aquire from characterstics of the sensor.
        If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_{1,k})
        '''
        expectedMeasurment:List = self._getCurrentExpectedMeasurment()
        actualMeasurment:List = self.getObservationsSerie().getLastObservation()
        mean:Vector=Vector([expectedMeasurment,actualMeasurment])
        covarianceMatrix:Matrix = None
        pdf = Gaussian(mean,covarianceMatrix)
        currentLikelihood = pdf.getValueByEvent(self.getObservationsSerie().getLastObservation())
        return currentLikelihood


    def _getCurrentMarginalLikelihood(self) -> float:
        ''''''
        pdf = Pdf()
        likelihoodSum: float = 0
        stateCounter = 0
        for state in self._stateSerie:
            likelihoodSum += pdf.getValueByEvent(
                Event(self.__observationSerie.getLastObservation()).conditionedOn(state)) \
                             * pdf.getValueByEvent(Event(state).conditionedOn(
                self.__observationSerie.getObservationSlice(0, self.__observationSerie.getLastObservationIndex() - 1)))
            stateCounter += 1
        self.__marginalLikelihood = likelihoodSum

    def getObservationsSerie(self)-> MeasurementsSerie:
        ''''''
        return self.__observationSerie

    def getStatesSerie(self)->StatesSerie:
        ''''''
        return self._stateSerie

    @abc.abstractmethod
    def _getCurrentExpectedMeasurment(self)->float:
        '''I think this should be found according the experience, ie the number of times an measurement was observed given teh state process'''




