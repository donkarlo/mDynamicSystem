from random import random

from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Particle import Particle
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.Decorator import Decorator
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.linearAlgebra.Vector import Vector
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Filter import Filter


class PolynomialDrawParticles(Decorator):
    def __init__(self,filter:Filter):
        self._filter = filter

    def _getNewParticleStateFromAPdf(self, predictedPriorParticle: Particle):
        return random.choice(self._intrestedRegion)

    def _getMeasurementLikelihoodByState(self, state: State):
        pass
