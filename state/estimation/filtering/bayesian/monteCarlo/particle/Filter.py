from random import random
from typing import List

from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.StateProbability import StateProbability
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.linear.MeasurementModel import MeasurementModel
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mMath.data.probability.Pdf import Pdf
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement.Measurement import Measurement
from sympy import DiracDelta

from mMath.data.probability.discrete.uniform.Uniform import Uniform
from mMath.linearAlgebra.Vector import Vector
import abc
from mMath.region.Discreet import Discreet


class Filter(MainFilter, abc.ABCMeta):
    '''
    - Both linear and nonlinear process and measurement models can be used
    '''

    def __init__(self
                 , stateSet:Discreet
                 , startingState:State
                 , processModel:ProcessModel
                 , measurementModel:MeasurementModel
                 , particlesNum:int):
        '''

        :param interestedRegion:matrix The region at which we want to know to what probability by which the agent is there
        :param particlesNum:int
        :param processNoiseCovarianceMatrix:matrix
        :param measuremetModel
        '''
        super().__init__(stateSet
                         , startingState
                         , processModel
                         , measurementModel)
        #particle related settings
        self._particlesNum = particlesNum

        # In-bulit application
        self._particles:List[Particle] = []
        self._initiateParticles()

    @abc.abstractmethod
    def _getNewParticleStateFromAPdf(slef, predictedPriorParticleState:State):
        pass

    @abc.abstractmethod
    def _getMeasurementLikelihoodByState(self, state: State):
        pass

    def _drawParticles(self):
        '''Take a sample from the predicted state'''
        loopingParticle: Particle
        for loopingParticle in self._particles:
            nextPredictedParticleState:State = self.getNextParticleStateWithoutNoise(loopingParticle)
            newParticleStateFromAPdf = self._getNewParticleStateFromAPdf(nextPredictedParticleState)
            newWeight = self._getNewWeightByNewStateAndPreviousWeight(newParticleStateFromAPdf, loopingParticle.getWeight())
            loopingParticle.update(nextPredictedParticleState, newWeight)


    def getNextParticleStateWithoutNoise(self,particle):
        self._processModel.update(particle.getState()
                                  , self._processModel.getCurrentControlInput()
                                  , self._processModel.getPreviousNoisePdf()
                                  , None)
        nextParticleState: State = self._processModel.getNextStateWithoutNoise()
        return nextParticleState

    def _initiateParticles(self):
        '''
        :return:
        '''
        sampleStates:List[State] = random.sample(self._stateSet.getSample())
        weight = 1 / self._particlesNum
        sampleState:Vector
        for sampleState in sampleStates:
            self._particles.append(Particle(sampleState, weight, self._measurementSerie.getLength()))

    def _onAddMeasurement(self, measurement:Measurement):
        '''

        :param measurement:
        :return:
        '''
        self._drawParticles()


    def subtractBaseDiracDelta(self,point:Vector,particle:Particle):
        '''

        :param point:
        :param particle:
        :return:
        '''
        return DiracDelta(point.getDistanceFrom(particle.getState()))

    def _updatePosteriors(self) ->float:
        '''
        - Updates posteriors af all points in the region of interest
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{k}-x^{i}_{k}), sum(w_k^i)=1
        '''
        state:State
        for state in self._stateSet:
            if self._particles is not None:
                sum = 0
                particle:Particle
                for particle in self._particles:
                    sum += particle.getWeight()*self.subtractBaseDiracDelta(state,particle.getState())
                statePosterior:statePosterior = StateProbability()
            self._posteriors.append(statePosterior)

    def _getNewWeightByNewStateAndPreviousWeight(self
                                                 , newState: State
                                                 , previousWeight: float):
        '''

        :param newStateRefVec:
        :param previousWeight:
        :return:
        '''

        newWeight = previousWeight * self._getMeasurementLikelihoodByState(newState)
        return newWeight

    def _getNextParticleStateWithoutNoise(self,previousParticle:Particle):
        return self._processModel.getNextStateWithoutNoise(previousParticle)
