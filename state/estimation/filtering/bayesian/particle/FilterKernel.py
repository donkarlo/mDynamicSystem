from typing import List

from mDynamicSystem.state.estimation.filtering.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.particle import ParticlePosVel
from mDynamicSystem.state.estimation.filtering.bayesian.particle.Particle import Particle
from mMath.linearalgebra.Matrix import Matrix
from mDynamicSystem.state import State
from mDynamicSystem.state.estimation import MeasurementEquation, ProcessModel


class FilterKernel(MainFilter):
    ''''''

    def __init__(self
                 , initState: State
                 , regionOfIntrestDistribution: Matrix
                 , particlesNum: int
                 ):
        '''
        @todo ROI can be a distribution too

        Parameters
        -----------

        '''
        self.__initState: State = initState
        self.__currentParticles = None
        self.__particlesNum = particlesNum
        self.__regionOfInterestDistribution = regionOfIntrestDistribution

        super().__init__()

    def initialize(self):
        pass

    # def __senseFunction(self, particle: particle)->float:
    #     '''z(x_t)~\sum f(x|mean(x),var(x)) and f=mMath.data.pdf.Normal(mean(x),var(x))
    #     measures particle's (position,velocity) likelihood based on observation
    #     x is the  measured distance between the particle and the given landmarke
    #     x_t is the particle
    #     \mu is the theoretical distance between the measured distance and the landmark
    #     '''
    #     pass

    def __generateCurrentParticles(self, size: int) -> None:
        ''''''
        pass

    def __getCurrentParticles(self) -> List[ParticlePosVel]:
        ''''''
        if self.__currentParticles is None:
            self.__generateCurrentParticles()
        return self.__currentParticles

    def moveParticlesTowardCurrentStateByStateEquation(self, stateEquation: ProcessModel):
        ''''''
        currentPredictedState: State = stateEquation.getCurrentState()
        particle: Particle
        for particle in self.__getCurrentParticles():
            moveRate: float = particle.getState().getDistanceFrom(currentPredictedState) / 100
            particle.getState().moveTowardVecByRate(currentPredictedState, moveRate)

    def getpPredictedObservationFromPredictedStateByMeasurementEquation(self
                                                                        , measurementEquation: MeasurementEquation
                                                                        , predictedState: State):
        ''''''
        pass

    def updateCurrentParticlesBelief(self):
        ''''''
        for particle in self.__currentParticles:
            pass

    def resampleCurrentParticlesAccordingImportanceWeightBelief(self):
        ''''''
        pass

    def computeBestParticleInCurrentParticles(self):
        ''''''
        pass

    def __getParticleWeight(self, particle) -> float:
        ''''''
        thisParticleKernel = self.__kernel0.getValue(
            particle.getMeasuredObservation() - particle.getPredictedObservation())

        kernelSum = 0
        particleInLoop: Particle
        for particleInLoop in self.__currentParticles:
            kernelSum += self.__kernel0.getValue(
                particleInLoop.__measuredObservation - particleInLoop.__predictedObservation)

        return thisParticleKernel / kernelSum

    def getEstimatedDensity(self, x):
        ''''''
        sum = 0
        particleInLoop: Particle
        for particleInLoop in self.__currentParticles:
            sum += self.__getParticleWeight(particleInLoop) * self.__kernel1.getValue(
                x - particleInLoop.getPredictedState())


