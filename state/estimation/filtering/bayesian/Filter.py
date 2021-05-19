import abc
from typing import List
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.calculus.SingleDefiniteIntegral import SingleDefiniteIntegral
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mMath.linearAlgebra.matrix.Matrix import Matrix
from mMath.linearAlgebra.Vector import Vector
from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.measurement.MeasurementsSerie import MeasurementsSerie


class Filter(Estimation):
    '''
    - Is an estimation as the following:
    - Bayesian inference allows for estimating a state by combining a statistical model for a measurement (likelihood)
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

    @abc.abstractmethod
    def _onAddMeasurement(self, observation: Measurement):
        '''

        :param observation:
        :return:
        '''
        pass

    @abc.abstractmethod
    def _getExpectedMeasurment(self) -> float:
        '''I think this should be found according the experience, ie the number of times an measurement was observed given teh state process'''
        pass


    def __init__(self):
        # Estimations improve by receiving more measurement
        self.__measurementsSerie: MeasurementsSerie = None
        #History of states
        self._statesSerie: StateSerie = None
        # p(x_k|z_{1:k-1})
        self._prior: float = None
        # p(x_k|z_{1:k}) = \frac{p(z_k|x_k)p(x_k|z_{1:k-1})}/{p(z_k|z_{1:k-1})}
        self._posterior = None
        # @see self._getMarginalLikelihood()
        self._marginalLikelihood: float = None

    def addMeasurement(self, measurement: Measurement) -> None:
        '''
        Anytime a new measurement is added, the the system can update its state(a random variable) belief (a PDF over
        the state random variable which quantifies uncertainty of being in that state)
        :param measurement: Measurement
        :return:
        '''
        self.__measurementsSerie.appendMeasurement(measurement)
        self._onAddMeasurement(measurement)
        self._updatePrior()
        self._updatePosterior()





    def getPosterior(self):
        '''
        :return:
        '''
        if self._posterior is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self.getPosterior")
        return self._posterior

    def _getPrior(self)->float:
        '''
        :return:
        '''
        if self._prior is None:
            raise Exception("Add a measuremt first using self.addMeasurement() before calling self._getPrior()")
        return self._prior

    def _updatePrior(self)->None:
        '''
        - The prior represents the best guess at time k given measurements up to time k − 1. It can be interpreted as the
            predicted state at time k.
        - integral('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1}dx_{k-1})',[-9999999999999,999999999999])
        - The integrals in (8) and (9) can only be solved analytically under strong assumptions, e.g., for finite
            dimensional discrete state variables or linear models and Gaussian pdfs.
        :return:
        '''
        self._prior = SingleDefiniteIntegral('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1})','dx_{k-1}',
                                             [-9999999999999,999999999999]).getValue()


    def _updatePosterior(self)->None:
        '''
        - This is the result we expect to recieve from any estimation
        - p(x_k|z_{1:k}) = (p(z_k|x_k)p(x_k|z_{1:k-1}))/(p(z_k|z_{1:k-1}))
        - Calculate p(x_{k}|u_{1:k},z_{1:k})
        - posterior = (likelihood.prior)/marginalLikelihood
        :return:
        '''
        self._posterior = self._getPrior() * ((self._getLikelihood()) / self._getMarginalLikelihood())

    def _updateLikelihood(self) -> float:
        '''A knowledge which we aquire from characterstics of the sensor.
        If expected expectedMeasurment coicides with actualMeasurment then the highest weight is gained
        p(z_k|x_{1,k})
        '''
        expectedMeasurment:List = self._getExpectedMeasurment()
        actualMeasurment:List = self.getMeasurementsSerie().getLastMeasurement()
        mean:Vector=Vector([expectedMeasurment,actualMeasurment])
        covarianceMatrix:Matrix = None
        pdf = Gaussian(mean,covarianceMatrix)
        self._likelihood = pdf.getValueByEvent(self.getMeasurementsSerie().getLastMeasurement())
        return self._likelihood

    def getLikelihood(self)->float:
        return self._likelihood


    def _updateMarginalLikelihood(self) -> float:
        '''
        p(z_k|z_{1:k-1})
        :return:
        '''
        self._prior = SingleDefiniteIntegral('p(z_k|x_{k})*p(x_{k}|z_{1:k-1})', 'dx_{k}',
                                             [-9999999999999, 999999999999]).getValue()

    def getMarginalLikelihood(self)->float:
        '''

        :return:
        '''
        return self._likelihood

    def getMeasurementsSerie(self)-> MeasurementsSerie:
        '''
        :return:
        '''
        return self.__measurementsSerie

    def getStatesSerie(self)->StatesSerie:
        '''
        :return:
        '''
        return self._statesSerie






