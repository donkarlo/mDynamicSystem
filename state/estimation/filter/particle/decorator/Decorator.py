from mDynamicSystem.state.estimation.filter.particle.Filter import Filter


class Decorator(Filter):
    def __init__(self,filter:Filter):
        self._filter = filter
