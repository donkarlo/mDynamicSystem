import abc
from typing import List

from mMath.data.probability.Event import Event
from mMath.data.probability.Pdf import Pdf
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.linearalgebra.Matrix import Matrix
from mMath.linearalgebra.Vector import Vector
from mDynamicSystem.state import State
from mDynamicSystem.state.Serie import Serie as StatesSerie
from mDynamicSystem.state.observation.Observation import Observation
from mDynamicSystem.state.observation.ObservationsSerie import ObservationsSerie


class Filter(metaclass=abc.ABCMeta):
    ''' In each filtering our goal is to compute the posterior which is state estimation after observation (self.getPosterior)
     first we predict () then we observe and update according to what is observed
    What is the PDF of state with a given set of observation
    if the variables are normally distributed and the transitions are linear, the Bayes filtering becomes equal to the Kalman filtering.
    '''

    def __init__(self):
        # Estimations improve by receiving more observation
        self.__observationSerie: ObservationsSerie = None
        #History of states
        self.__stateSerie: StatesSerie = None
        # p(x_k|z_{1:k-1})
        self.__currentPrior: float = None
        # Acts as a normalizer, a constant which could be regarded as the probability of the measurement, sum(p(z_{1:k}))
        self.__currentMarginalLikelihood: float = None

        self.__stateSerie = None

    def addObservation(self, observation: Observation) -> None:
        '''Whenever a new observation is added prediction and then update should run again'''
        self.__observationSerie.appendObservation(observation)
        self._predict()
        self._update()

    def _predict(self) -> State:
        '''First setp: predict prior to observation using the StateEquation'''
        self._updatePrior()
        self._updateLikelihood()

    @abc.abstractmethod
    def _update(self, observationSerie: Observation) -> State:
        '''Second step: Update prediction, refining the _pridct sing bayes theorem'''
        pass

    def getCurrentPosterior(self):
        '''No stop condition is needed. the last observation is the s'''
        currentPosterior = (self._getLikelihood() * self._getCurrentPrior()) / self._getMarginalLikelihood()
        return currentPosterior

    def _getCurrentPrior(self) -> float:
        ''''''
        pdf = Pdf()
        priorSum: float = 0
        stateCounter = 0
        for state in self.__stateSerie:
            priorSum += pdf.getValueByEvent(
                Event(state).conditionedOn(Event(self.__stateSerie.getByIndex(stateCounter - 1))))
            stateCounter += 1
        self.__currentPrior = priorSum
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
        for state in self.__stateSerie:
            likelihoodSum += pdf.getValueByEvent(
                Event(self.__observationSerie.getLastObservation()).conditionedOn(state)) \
                             * pdf.getValueByEvent(Event(state).conditionedOn(
                self.__observationSerie.getObservationSlice(0, self.__observationSerie.getLastObservationIndex() - 1)))
            stateCounter += 1
        self.__currentMarginalLikelihood = likelihoodSum

    def getObservationsSerie(self)->ObservationsSerie:
        ''''''
        return self.__observationSerie

    def getStatesSerie(self)->StatesSerie:
        ''''''
        return self.__stateSerie

    @abc.abstractmethod
    def _getCurrentExpectedMeasurment(self)->float:
        '''I think this should be found according the experience, ie the number of times an observation was observed given teh state process'''




