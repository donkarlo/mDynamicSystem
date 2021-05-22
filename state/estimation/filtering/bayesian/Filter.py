import abc
from typing import List

from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mMath.linearAlgebra.matrix.Matrix import Matrix
from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.measurement.MeasurementSerie import MeasurementSerie
from mDynamicSystem.state.estimation.filtering.bayesian.StateProbability import StateProbability
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel

class Filter(Estimation,abc.ABCMeta):
    '''
    - Is an estimation as the following:
    - Bayesian inference allows for estimating a state by combining a statistical processModel for a measurement (likelihood)
        with a prior probability using Bayes’ theorem.
    - Byesian Filtering is a form of Estimation in which the intersection of measurement distribution and predicted, next
        state distribuation makes restricts the answer set
    - Byesian Filter refine estimates everytime a new measurement is achieved
    - In each Byesian Filter our goal is to compute the posterior which is state estimation after measurement (self.getPosterior)
    - first we predict () then we observe and update according to what is observed
    - What is the PDF of state with a given set of measurement
    - if the variables are normally distributed and the transitions are linear, the Bayes filtering becomes equal to
        the Kalman filtering.
    '''

    def __init__(self
                 ,intrestedRegion:Matrix
                 ,startingState:State
                 ,processModel:ProcessModel
                 ,measurementModel:MeasurementModel
                 ):
        #
        self._intrestedRegion:Matrix = intrestedRegion
        #
        self._startingState:Matrix = startingState
        #
        self._processModel:ProcessModel = processModel
        #
        self._measurementModel:MeasurementModel = measurementModel
        # Estimations improve by receiving more measurement
        self._measurementSerie: MeasurementSerie = None
        #History of states
        self._stateSerie: StateSerie = None
        self._stateSerie.appendState(self._startingState)
        # p(x_k|z_{1:k-1})
        self._intrestedRegionStatePriors: List[StateProbability] = []
        # p(x_k|z_{1:k}) = \frac{p(z_k|x_k)p(x_k|z_{1:k-1})}/{p(z_k|z_{1:k-1})}
        # Posterior should hold the points in region of intrest plus the probability that agent is in each of those points
        # To store the probabilities by which the agent may reside in each region in region of interest
        self._interestedRegionPosteriors: List[StateProbability] = []
        #
        self._intrestedRegionMeasurementsLikelihoods:List = None
        # @see self._getMarginalLikelihood()
        self._marginalMeasurementsLikelihood: float = None

    @abc.abstractmethod
    def _onAddMeasurement(self, measurement: Measurement):
        '''

        :param measurement:
        :return:
        '''
        pass

    @abc.abstractmethod
    def _getExpectedMeasurment(self) -> float:
        '''I think this should be found according the experience, ie the number of times an measurement was observed given teh state process'''
        pass

    @abc.abstractmethod
    def _updateMarginalLikelihood(self):
        '''A knowledge which we aquire from characterstics of the sensor.
        If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_{1,k})
        '''
        pass

    @abc.abstractmethod
    def _updateIntrestedRegionStatePriors(self)->None:
        '''
        - The prior represents the best guess at time k given measurements up to time k − 1. It can be interpreted as the
            predicted state at time k.
        - integral('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1}dx_{k-1})',[-9999999999999,999999999999])
        - The integrals in (8) and (9) can only be solved analytically under strong assumptions, e.g., for finite
            dimensional discrete state variables or linear models and Gaussian pdfs.
        :return:
        '''

    @abc.abstractmethod
    def _updateIntrestedRegionStatePosteriors(self)->None:
        '''
        - This is the result we expect to recieve from any estimation
        - p(x_k|z_{1:k}) = (p(z_k|x_k)p(x_k|z_{1:k-1}))/(p(z_k|z_{1:k-1}))
        - Calculate p(x_{k}|u_{1:k},z_{1:k})
        - posterior = (likelihood.prior)/marginalLikelihood
        - Updates posteriors af all points in the region of interest
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{k}-x^{i}_{k}), sum(w_k^i)=1
        :return:
        '''
        pass

    @abc.abstractmethod
    def _updateIntrestedRegionLikelihoods(self) -> float:
        '''A knowledge which we aquire from characterstics of the sensor.
        If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_{1,k})
        '''




    def addMeasurement(self, measurement: Measurement) -> None:
        '''
        Anytime a new measurement is added, the the system can update its state(a random variable) belief (a PDF over
        the state random variable which quantifies uncertainty of being in that state)
        :param measurement: Measurement
        :return:
        '''
        self._measurementSerie.appendMeasurement(measurement)
        self._onAddMeasurement(measurement)
        self._updateIntrestedRegionMarginalLikelihood()
        self._updateIntrestedRegionLikelihoods()
        self._updateIntrestedRegionStatePriors()
        self._updateIntrestedRegionStatePosteriors()


    def getStatePosteriorProbabilities(self):
        '''
        :return:
        '''
        if self._statePosteriorProbabilities is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self.getPosterior")
        return self._statePosteriorProbabilities

    def _getIntrestedRegionStatePriors(self)->float:
        '''
        :return:
        '''
        if self._intrestedRegionStatePriors is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self._getPrior()")
        return self._intrestedRegionStatePriors

    def getIntrestedRegionMeasurementLikelihoods(self)->float:
        return self._intrestedRegionMeasurementsLikelihoods

    def getMarginalMeasurementsLikelihood(self)->float:
        '''

        :return:
        '''
        return self._marginalMeasurementsLikelihood

    def getMeasurementsSerie(self)-> MeasurementSerie:
        '''
        :return:
        '''
        return self._measurementSerie

    def getStatesSerie(self)->StateSerie:
        '''
        :return:
        '''
        return self._stateSerie






