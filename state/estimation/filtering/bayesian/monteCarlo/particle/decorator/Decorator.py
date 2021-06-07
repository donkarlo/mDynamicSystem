from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Filter import Filter
import abc

class Decorator(Filter):
    def __init__(self,filter:Filter):
        self._filter = filter
