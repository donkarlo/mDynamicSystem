import abc
from typing import List
from mDynamicSystem.state.estimation.Estimation import Estimation
from mMath.calculus.SingleDefiniteIntegral import SingleDefiniteIntegral
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.linearAlgebra.matrix.Matrix import Matrix
from mMath.linearAlgebra.Vector import Vector

class ConceretFilter(Estimation,abc.ABCMeta):
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


    def __init__(self, intrestedRegion:Matrix, startingState:State):
        super().__init__(intrestedRegion,startingState)

    def _updatePriors(self)->None:
        self._intrestedRegionStatePriors = SingleDefiniteIntegral('p(x_k|x_{k-1})*p(x_{k-1}|z_{1:k-1})', 'dx_{k-1}',
                                                   [-9999999999999,999999999999]).getValue()


    def _updatePosteriors(self)->None:
        self._interestedRegionPosteriors = self._getIntrestedRegionStatePriors() * ((self._updateIntrestedRegionLikelihoods) / self._getMarginalLikelihood())

    def _updateMarginalLikelihood(self) -> float:
        expectedMeasurment:List = self._getExpectedMeasurment()
        actualMeasurment:List = self.getMeasurementsSerie().getLastMeasurement()
        mean:Vector=Vector([expectedMeasurment,actualMeasurment])
        covarianceMatrix:Matrix = None
        pdf = Gaussian(mean,covarianceMatrix)
        self._intrestedRegionMarginalLikelihood = pdf.getValueAt(self.getMeasurementsSerie().getLastMeasurement())

    def getLikelihood(self)->float:
        return self._likelihood


    def _updateMarginalLikelihood(self) -> float:
        '''
        p(z_k|z_{1:k-1})
        :return:
        '''
        self._statePriors = SingleDefiniteIntegral('p(z_k|x_{k})*p(x_{k}|z_{1:k-1})', 'dx_{k}',
                                                   [-9999999999999, 999999999999]).getValue()








