import abc
from typing import List

from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.data.probability.Pdf import Pdf
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.measurement.Serie import Serie
from mDynamicSystem.state.estimation.filtering.bayesian.StateProbability import StateProbability
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.data.timeSerie.stochasticProcess.state.StateSet import StateSet


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
                 , stateSet:StateSet
                 , startingState:State
                 , processModel:ProcessModel
                 , measurementModel:MeasurementModel
                 ):
        #
        self._stateSet:StateSet = stateSet
        #
        self._startingState = startingState
        #
        self._processModel:ProcessModel = processModel
        #
        self._measurementModel:MeasurementModel = measurementModel
        # Estimations improve by receiving more measurement
        self._measurementSerie: Serie = None
        #History of states
        self._stateSerie: StateSerie = None
        self._stateSerie.appendState(self._startingState)
        # p(x_k|z_{1:k-1})
        self._priors: List[StateProbability] = []
        # p(x_k|z_{1:k}) = \frac{p(z_k|x_k)p(x_k|z_{1:k-1})}/{p(z_k|z_{1:k-1})}
        # Posterior should hold the points in region of intrest plus the probability that agent is in each of those points
        # To store the probabilities by which the agent may reside in each region in region of interest
        self._posteriors: List[StateProbability] = []
        #
        self._measurementsLikelihoods:List = None
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
    def _updatePriors(self)->None:
        '''
        - Predict probability of presence in every state set member at a time instant
        - The prior represents the best guess at time k given measurements up to time k − 1. It can be interpreted as the
            predicted state at time k.
        - integral('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1}dx_{k-1})',[-9999999999999,999999999999])
        - The integrals in (8) and (9) can only be solved analytically under strong assumptions, e.g., for finite
            dimensional discrete state variables or linear models and Gaussian pdfs.
        :return:
        '''
        pass

    @abc.abstractmethod
    def _updatePosteriors(self) -> None:
        '''
        - Update  probability of presence in every state set member at a time instant
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
    def _updateLikelihoods(self) -> float:
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

        #prediction phase using process model: the next state probabilities for intrested rgion points based on previous state using the process model
        #Such as transition matrix
        self._updatePriors()
        #update phase: refining the prediction phase prediction  using the new observation
        self._updatePosteriors()

    def _updateMarginalLikelihood(self):
        '''A knowledge which we aquire from characterstics of the sensor.
        If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_{1,k})
        '''
        pass


    def getPosteriors(self):
        '''
        :return:
        '''
        if self._posteriors is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self.getPosterior")
        return self._posteriors

    def _getPriors(self)->float:
        '''
        :return:
        '''
        if self._priors is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self._getPrior()")
        return self._priors

    def getMeasurementLikelihoods(self)->float:
        return self._measurementsLikelihoods

    def getMarginalMeasurementsLikelihood(self)->float:
        '''

        :return:
        '''
        return self._marginalMeasurementsLikelihood

    def getMeasurementsSerie(self)-> Serie:
        '''
        :return:
        '''
        return self._measurementSerie

    def getStatesSerie(self)->StateSerie:
        '''
        :return:
        '''
        return self._stateSerie






