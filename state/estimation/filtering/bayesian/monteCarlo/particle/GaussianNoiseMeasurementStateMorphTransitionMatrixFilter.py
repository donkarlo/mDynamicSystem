from typing import List

from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.filtering.bayesian.StateProbability import StateProbability
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Filter import Filter as ParticleFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.process import GaussianNoiseStateTransitionMatrixModel as GaussianNoiseStateTransitionProcessMatrixProcessModel

from mDynamicSystem.state.measurement.GaussianNoiseStateMorphModel import GaussianNoiseStateMorphModel as GaussianNoiseStateMorphMeasurementModel
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.linearAlgebra.Vector import Vector
from mMath.linearAlgebra.matrix import Matrix


class GaussianNoiseMeasurementStateMorphTransitionMatrixFilter(ParticleFilter):
    ''''''
    def __init__(self
                 , intrestedRegion:Matrix
                 , startingState:State
                 , gaussianNoiseStateTransitionMatrixProcessModel:GaussianNoiseStateTransitionProcessMatrixProcessModel
                 , gaussianNoiseStateMorphMeasurementModel:GaussianNoiseStateMorphMeasurementModel
                 ):
        '''

        :param gaussianNoiseStateTransitionMatrixProcessModel:
        :param gaussianNoiseStateMorphMeasurementModel
        '''
        super().__init__(intrestedRegion,startingState)
        self._gaussianNoiseStateTransitionMatrixProcessModel:GaussianNoiseStateTransitionProcessMatrixProcessModel = gaussianNoiseStateTransitionMatrixProcessModel
        self._gaussianNoiseStateMorphMeasurementModel:GaussianNoiseStateMorphMeasurementModel = gaussianNoiseStateMorphMeasurementModel

    def _darwParticles(self):
        '''
        :return:
        '''
        particle:Particle
        for particle in self._particles:
            nextParticleState: State = self._gaussianNoiseStateTransitionMatrixProcessModel.getNextState(
                self.particle.getState())
            gaussian = Gaussian(nextParticleState.getRefVec(),
                                self._gaussianNoiseStateTransitionMatrixProcessModel.getGaussianPdf().getCovariance())
            newParticleState:State = State(Vector(gaussian.getASample()))
            # self._gaussianNoiseStateMorphMeasurementModel.updateState(newParticleState)
            # expectedState = State(self._gaussianNoiseStateMorphMeasurementModel.getMeasurementWithoutNoise().getRefVec())
            newWeight = self._getNewWeightByNewStateAndPreviousWeight(newParticleState, particle.getWeight())
            particle.update(newParticleState,newWeight)

    def _getNewWeightByNewStateAndPreviousWeight(self
                                                 , newState:State
                                                 , previousWeight:float):
        '''

        :param newStateRefVec:
        :param previousWeight:
        :return:
        '''
        
        newWeight = previousWeight*self._getLikelihoodByState(newState)
        return newWeight
    
    def _getLikelihoodByState(self,state:State)->float:
        gaussian: Gaussian = Gaussian(self._measurementSerie.getLastMeasurement()
                                      , self._gaussianNoiseStateMorphMeasurementModel.getGaussianPdf().getCovariance())
        likelihood = gaussian.getValueByEvent(self._gaussianNoiseStateMorphMeasurementModel.getMeasurementWithoutNoise(state))
        return likelihood



    def _updateStatePosteriors(self) ->float:
        '''
        - Updates posteriors af all points in the region of interest
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{k}-x^{i}_{k}), sum(w_k^i)=1
        '''
        stateInRegionOfInterest:State
        regionOfStatesProbabilities = []
        for stateInRegionOfInterest in self._intrestedRegionStates:
            if self._particles is not None:
                sum = 0
                particle:Particle
                for particle in self._particles:
                    sum += particle.getWeight()*self.subtractBaseDiracDelta(stateInRegionOfInterest,particle.getState())
                statePosterior:statePosterior = StateProbability()
            regionOfStatesProbabilities.append(statePosterior)