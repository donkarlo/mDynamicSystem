import abc
from mDynamicSystem.state.estimation.process.Model import Model

class Decorator(Model,abc.ABCMeta):
    '''
    - Inherit from model just to enforce kids to have getCurrentMeasurements of their own
    '''
    def __init__(self,model:Model):
        self._model:Model = model