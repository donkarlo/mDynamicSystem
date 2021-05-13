from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mMath.data.probability.Event import Event
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.data.probability.discrete.Pdf import Pdf
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement.Measurement import Measurement


class Filter(MainFilter):
    def __init__(self,numberOfParticles:int):
        ''''''
        self.__numberOfParticles = numberOfParticles
        self.__particles = None
        self.__currentMeasurement:Measurement = None
        self.__previousMeasurement:Measurement = None

    def _onAddMeasurement(self, measurement:Measurement):
        self.__previousMeasurement = self.__currentMeasurement
        self.__currentMeasurement = measurement


    def _getParticleWeightByParticle(self, particle: Particle):
        ''''''
        pdf = Pdf()
        lastMeasurementEvent:Measurement = Event(self.getObservationsSerie().getLastObservation())
        lastStateEvent:State = Event(self.getStatesSerie().getLastState())
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

    def getPosterior(self) ->float:
        if self.__particles is not None:
            sum = 0
            particle:Particle
            for particle in self.__particles:
                sum += particle.getWeight()*self.dirac(self._statesSerie.getLastState()-particle.getState())
        return sum

    def dirac(self,x):
        if x==0:
            return 99999999999999999999999999
        return 0







