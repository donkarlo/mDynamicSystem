from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreetLevel.SuperState import SuperState
from mDynamicSystem.state.measurement.Model import Model as MainMeasurementModel

class MeasurementModel(MainMeasurementModel):
    '''Should convert a superstate to a measurement'''
    def __init__(self,superState:SuperState):
        super().__init__(superState,None,None)

    def getMeasurementWithoutNoise(self) -> Measurement:
        return Measurement(self._state.getRefVec())

