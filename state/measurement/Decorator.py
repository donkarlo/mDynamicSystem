from mDynamicSystem.state.measurement.Model import Model
import abc


class Decorator(Model,abc.ABCMeta):
    '''
    - Inherit from model just to enforce kids to have getCurrentMeasurements of their own
    '''
    def __init__(self,model:Model):
        self._model:Model = model