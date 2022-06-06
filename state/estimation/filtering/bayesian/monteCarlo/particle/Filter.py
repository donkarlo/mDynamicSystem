from typing import List
from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.linear.MeasurementModel import MeasurementModel
from mDynamicSystem.state.estimation.process.Model import Model as ProcessModel
from mDynamicSystem.state.State import State
from mDynamicSystem.obs.Obs import Obs
from sympy import DiracDelta
from mMath.data.probability.Pdf import Pdf
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.StateSpace import StateSpace
import abc


class Filter(MainFilter, metaclass=abc.ABCMeta):
    '''
    - Both linear and nonlinear process and obs models can be used
    '''

    def __init__(self
                 , particlesNum: int
                 , stateSpace: StateSpace
                 , processModel: ProcessModel
                 , measurementModel: MeasurementModel
                 , startingState:State
                 ):
        '''

        :param particlesNum:
        :param stateSpace:
        :param processModel:
        :param measurementModel:
        :param predictedStateMeasurementLikelihood:
        :param startingState:
        '''
        super().__init__(stateSpace
                         , startingState
                         , processModel
                         , measurementModel
                         )
        #particle related settings
        self._particlesNum:int = particlesNum
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

    def _updateParticles(self):
        '''
        - Draw a new sample from each particle
        - Take a sample from the predicted state
        '''
        loopingParticle: Particle
        for loopingParticle in self._particles:
            #first move the particle state along the process model
            self._updateParticle(loopingParticle)

    def updateParticle(self,particle:Particle)->None:
       particle.updateState(self._getPredictedState(particle.getState()))
       particle.updateWeight(self._getUpdatedParticleWeight(particle.getState(),particle.getWeight()))

    def _getPredictedState(self, state:State)->State:
        '''

        :param particle:
        :return:
        '''
        predictedParticleState:State = self._processModel.getPredictedState2(state)
        return predictedParticleState

    def _getUpdatedParticleWeight(self, particleState:State,previousParticleWeight:float)->float:
        '''
        - U[dating the likelihood of a particle whenever a new obs is available
        :return:
        '''
        gaussianPdf:Pdf = Gaussian(self.getMeasurementsSerie().getLastMeasurement().getRefVec()
                               ,self.getMeasurementModel().getNoisePdf().getCovarianceMatrix())
        newWeight = previousParticleWeight * gaussianPdf.getValueAt(self.__getParticleStateMeasurement(particleState))
        return newWeight



    def __getParticleStateMeasurement(self, state:State)->Obs:
        # convert the predicted state to predicted obs
        self._measurementModel.updateState(state)
        measurement: Obs = self._measurementModel.getMeasurement()
        return measurement

    def _getStatePosterior(self,state:State) ->float:
        '''
        - Updates posteriors af all points in the region of interest
        - particle filter approximates the pdf representing the posterior by a discrete pdf such that there are minimal
            restrictions on the models involved. The optimal Bayesian solution is approximated by a sum of weighted
            samples:
        - p(x_{0:k}|z_{1:k}) = sum_{1}^{N_s}w^{i}_{k}dirac(x_{k}-x^{i}_{k}), sum(w_k^i)=1
        '''
        self._updateParticles()
        if self._particles is not None:
            sum = 0
            particle:Particle
            for particle in self._particles:
                sum += particle.getWeight()*DiracDelta(state.getRefVec()-particle.getState().getRefVec())
        return sum

