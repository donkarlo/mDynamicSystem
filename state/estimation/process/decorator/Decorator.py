from mDynamicSystem.state.estimation.process.Model import Model


class Decorator(Model):
    def __init__(self,model:Model):
        self._model = model