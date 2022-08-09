from mDynamicSystem.state.estimation.filter.particle.decorator.Decorator import Decorator
from mMath.probability.continous.gaussian.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.linearAlgebra.Vector import Vector


class GaussianDrawParticles(Decorator):

    def _getNewParticleStateFromAPdf(self, priorParticleState:State)->State:
        '''
        :return:
        '''
        gaussian = Gaussian(priorParticleState.getRefVec(), self._processModel.getCovariance())
        newParticleState: State = State(Vector(gaussian.getASample()))
        return newParticleState
