from typing import List

from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.ProcessModel import ProcessModel
from mDynamicSystem.state.estimation.linear.MeasurementModel import MeasurementModel
from mMath.data.probability.continous.uniform.Uniform import Uniform
from mMath.data.probability.event.Event import Event
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.data.probability.discrete.Pdf import Pdf
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement.Measurement import Measurement
from sympy import DiracDelta
from mMath.linearAlgebra.Vector import Vector
import abc

from mMath.linearAlgebra.matrix.Matrix import Matrix


class Filter(MainFilter, abc.ABCMeta):
    '''
    - Both linear and nonlinear process and measurement models can be used
    '''
    @abc.abstractmethod
    def _drawParticles(self):
        '''Take a sample from the predicted state'''
        pass
    def __init__(self
                 , intrestedRegion:Matrix
                 , startingState:State
                 , processModel:ProcessModel
                 , measurementModel:MeasurementModel
                 , particlesNum:int):
        '''

        :param interestedRegion:Matrix The region at which we want to know to what probability by which the agent is there
        :param particlesNum:int
        :param processNoiseCovarianceMatrix:matrix
        :param measuremetModel
        '''
        super().__init__(intrestedRegion
                         ,startingState
                         ,processModel
                         ,measurementModel)
        #particle related settings
        self._particlesNum = particlesNum

        # In-bulit application
        self._particles:List[Particle] = []
        self._initiateParticles()

    def _initiateParticles(self):
        samples: Matrix = Uniform.getSamples(self._particlesNum)
        weight = 1 / self._particlesNum
        for sampleNpRow in samples.getNpRows():
            sampleVec = Vector(sampleNpRow)
            self._particles.append(Particle(sampleVec, weight, self._measurementSerie.getLength()))

    def _onAddMeasurement(self, measurement:Measurement):
        self._drawParticles()

    def _getParticleWeightByParticle(self, particle: Particle):
        '''

        :param particle:
        :return:
        '''
        pdf = Pdf()
        lastMeasurementEvent:Measurement = Event(self.getMeasurementsSerie().getLastMeasurement())
        lastStateEvent:State = Event(self.getStatesSerie().getStateEvents())
        updatedWeight = particle.getWeight() * pdf.getValueByEvent(lastMeasurementEvent.conditionedOn(lastStateEvent))
        return updatedWeight

    def updateParticleWeightByMeasurement(self, particle:Particle, measurement:Measurement)->None:
        '''w^{i}_{k}^i ~ w^{i}_{k-1}p(z_k|x^{i}_{k})
        :param particle:
        :param measurement:
        :return:
        '''
        newWeight = particle.getWeight()*self.getMeasurementLikelihoodByState(measurement,particle.getState())
        particle.updateWeight(newWeight)

    def getMeasurementLikelihoodByState(self, measurement:Measurement, state:State)->Gaussian:
        '''N(z^{^},z,R)
        :param measurement:
        :param state:
        :return:
        '''
        predictedStateByParticleState = self.getPredictedMeasurementByState(state)
        measurementLikelihoodByStateGaussian:Gaussian = Gaussian([predictedStateByParticleState,measurement])
        return measurementLikelihoodByStateGaussian

    def getPredictedMeasurementByState(self,state:State)->float:
        pass

    def subtractBaseDiracDelta(self,point:Vector,particle:Particle):
        '''

        :param point:
        :param particle:
        :return:
        '''
        return DiracDelta(point.getDistanceFrom(particle.getState()))

    def _updateIntrestedRegionMarginalLikelihood(self) -> float:
        '''
        :return:
        '''
        pdf:Pdf = Pdf()
        likelihoodSum: float = 0
        stateCounter = 0
        for state in self._stateSerie:
            likelihoodSum += pdf.getValueByEvent(
                Event(self._measurementSerie.getLastMeasurement()).conditionedOn(state)) \
                             * pdf.getValueByEvent(Event(state).conditionedOn(
                self._measurementSerie.getMeasurementSlice(0, self._measurementSerie.getLastMeasurementIndex() - 1)))
            stateCounter += 1
        self._marginalLikelihood = likelihoodSum







