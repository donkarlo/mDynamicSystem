from random import random
from typing import List

from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.StateProbability import StateProbability
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.linear.MeasurementModel import MeasurementModel
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement.Measurement import Measurement
from sympy import DiracDelta

from mMath.data.probability.Pdf import Pdf
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
from mMath.linearAlgebra.Vector import Vector
import abc


class Filter(MainFilter, abc.ABCMeta):
    '''
    - Both linear and nonlinear process and measurement models can be used
    '''

    def __init__(self
                 , particlesNum: int
                 , stateSpace: StateSpace
                 , processModel: ProcessModel
                 , measurementModel: MeasurementModel
                 , measurementStateLikelihoodPdf:Pdf
                 , startingState:State
                 ):
        '''

        :param particlesNum:
        :param stateSpace:
        :param processModel:
        :param measurementModel:
        :param measurementStateLikelihoodPdf:
        :param startingState:
        '''
        super().__init__(stateSpace
                         , startingState
                         , processModel
                         , measurementModel
                         )
        #particle related settings
        self._particlesNum:int = particlesNum
        self._measurementStateLikelihoodPdf:Pdf = measurementStateLikelihoodPdf
        # In-bulit application
        self._particles:List[Particle] = []
        self._initiateParticles()

    def _initiateParticles(self):
        '''
        :return:
        '''
        sampleStates:List[State] = self._stateSpace.getSample(self._particlesNum)
        weight = 1 / self._particlesNum
        sampleState:State
        for sampleState in sampleStates:
            self._particles.append(Particle(sampleState, weight, self._measurementSerie.getLength()))

    def _drawParticles(self):
        '''Take a sample from the predicted state'''
        loopingParticle: Particle
        for loopingParticle in self._particles:
            #first move the particle state along the process model
            predictedParticleState:State = self._processModel.getPredictedState(loopingParticle.getState())

            #convert the predicted state to predicted measurement
            self._measurementModel.updateState(predictedParticleState)
            predictedMeasurement:Measurement = self._measurementModel.getMeasurement()

            #the new weight
            newWeight = loopingParticle.getWeight() * self._measurementStateLikelihoodPdf.getValueAt(predictedMeasurement.getRefVec())
            #get new weight
            loopingParticle.update(predictedParticleState, newWeight)


    def _predictStateSpacePriorProbabilities(self):
        '''
        is implemented in self._drawParticles() which is called by _updateStateSpacePosteriorProbabilities
        :return:
        '''
        pass


    def subtractBaseDiracDelta(self,point:Vector,particle:Particle):
        '''

        :param point:
        :param particle:
        :return:
        '''
        return DiracDelta(point.getDistanceFrom(particle.getState()))


    def _updateStateSpacePosteriorProbabilities(self):
        '''
        - Updates posteriors af all points in the region of interest
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{k}-x^{i}_{k}), sum(w_k^i)=1
        '''
        self._drawParticles()
        state:State
        for state in self._stateSpace:
            if self._particles is not None:
                sum = 0
                particle:Particle
                for particle in self._particles:
                    sum += particle.getWeight()*self.subtractBaseDiracDelta(state,particle.getState())
                statePosterior:statePosterior = StateProbability()
            self._stateSpacePosteriorProbabilities.append(statePosterior)

