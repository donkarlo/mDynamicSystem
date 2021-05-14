from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mMath.data.probability.Event import Event
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.data.probability.discrete.Pdf import Pdf
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement.Measurement import Measurement
from mMath.linearalgebra.Matrix import Matrix


class Filter(MainFilter):
    '''
    - both linear and nonlinear process and measurement models can be used
    '''
    def __init__(self,numberOfParticles:int,processNoiseCovarianceMatrix:Matrix,measuremetNoiseCovarianceMatrix:Matrix):
        '''

        :param numberOfParticles:int
        :param processNoiseCovarianceMatrix:Matrix
        :param measuremetNoiseCovarianceMatrix
        '''
        #particle related settings
        self.__numberOfParticles = numberOfParticles

        #sensor related settings
        self.__processNoiseCovarianceMatrix:Matrix = processNoiseCovarianceMatrix
        self.__measuremetNoiseCovarianceMatrix:Matrix = measuremetNoiseCovarianceMatrix

        #
        self.__particles = None
        self.__currentMeasurement:Measurement = None
        self.__previousMeasurement:Measurement = None

    def _onAddMeasurement(self, measurement:Measurement):
        self.__previousMeasurement = self.__currentMeasurement
        self.__currentMeasurement = measurement


    def _getParticleWeightByParticle(self, particle: Particle):
        ''''''
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

    def _updatePosterior(self) ->float:
        '''
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{0:k}-x_^{i}_{0:k}), sum(w_k^i)=1
        '''
        if self.__particles is not None:
            sum = 0
            particle:Particle
            for particle in self.__particles:
                sum += particle.getWeight()*self.dirac(self._statesSerie.getStateEvents() - particle.getState())
        return sum

    def dirac(self,x):
        if x==0:
            return 99999999999999999999999999
        return 0

    def _updateMarginalLikelihood(self) -> float:
        '''
        :return:
        '''
        pdf:Pdf = Pdf()
        likelihoodSum: float = 0
        stateCounter = 0
        for state in self._statesSerie:
            likelihoodSum += pdf.getValueByEvent(
                Event(self.__measurementsSerie.getLastMeasurement()).conditionedOn(state)) \
                             * pdf.getValueByEvent(Event(state).conditionedOn(
                self.__measurementsSerie.getObservationSlice(0, self.__measurementsSerie.getLastObservationIndex() - 1)))
            stateCounter += 1
        self._marginalLikelihood = likelihoodSum







