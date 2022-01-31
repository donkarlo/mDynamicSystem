import abc
from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.obs.Serie import Serie
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace


class Filter(Estimation,metaclass=abc.ABCMeta):
    '''
    - Is an estimation as the following:
    - Bayesian inference allows for estimating a state by combining a statistical processModel for a obs (likelihood)
        with a prior probability using Bayes’ theorem.
    - Byesian Filtering is a form of Estimation in which the intersection of obs distribution and predicted, next
        state distribuation makes restricts the answer set
    - Byesian Filter refine estimates everytime a new obs is achieved
    - In each Byesian Filter our goal is to compute the posterior which is state estimation after obs (self.getPosterior)
    - first we predict () then we observe and update according to what is observed
    - What is the PDF of state with a given set of obs
    - if the variables are normally distributed and the transitions are linear, the Bayes filtering becomes equal to
        the Kalman filtering.
    '''

    def __init__(self
                 , stateSpace: StateSpace
                 , startingState:State
                 , processModel:ProcessModel
                 , measurementModel:MeasurementModel
                 ):
        self._stateSpace = stateSpace
        #
        self._startingState = startingState
        #
        self._processModel:ProcessModel = processModel
        #
        self._measurementModel:MeasurementModel = measurementModel
        # Estimations improve by receiving more obs
        self._measurementSerie: Serie = None
        #History of highest probability(Best guess) states
        self._stateSerie: StateSerie = StateSerie()
        self._stateSerie.appendState(self._startingState)

    @abc.abstractmethod
    def getMaximumStatePosterior(self)->State:
        pass

    def _getStatePrior(self, currentState:State)->float:
        '''
        - next state prediction using the process model
        - Predict probability of presence in every state set member at a time instant
        - The prior represents the best guess at time k given measurements up to time k − 1. It can be interpreted as the
            predicted state at time k.
        - \int('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1}dx_{k-1})',[-\infty,+\infty])
        - The integrals in (8) and (9) can only be solved analytically under strong assumptions, e.g., for finite
            dimensional discrete state variables or linear models and Gaussian pdfs.
        :return:
        '''
        pass

    def _getLikelihood(self) -> float:
        '''
        - Last obs likelihood conditioned on predicted state by process model
        - A knowledge which we aquire from characterstics of the sensor.
        - In PF, If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_k), x_k is best guessed state
        '''
        pass
    def _getStatePosterior(self,state:State)->float:
        '''
        - Update  probability of presence in every state in state space at a time instant after having the obs and the neigbourhood its noise makes for it
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
        prior = self._getStatePrior(state)
        likelihood = self._getMeasurementLikelihood(self._measurementSerie.getLastMeasurement())
        posterior:float = (prior*likelihood)/self._getMeasurementMarginalLikelihood()
        return posterior





    def appendMeasurement(self, measurement: Obs) -> None:
        '''
        Anytime a new obs is added, the the system can update its state(a random variable) belief (a PDF over
        the state random variable which quantifies uncertainty of being in that state)
        :param measurement: Measurement
        :return:
        '''
        self._measurementSerie.appendMeasurement(measurement)


    def _getMeasurementMarginalLikelihood(self, measurement:Obs)->float:
        '''

        :return:
        '''
        pass

    def getMeasurementsSerie(self) -> Serie:
        '''
        :return:
        '''
        return self._measurementSerie

    def getStatesSerie(self)->StateSerie:
        '''
        :return:
        '''
        return self._stateSerie

    def getMeasurementModel(self)->MeasurementModel:
        return self._measurementModel






