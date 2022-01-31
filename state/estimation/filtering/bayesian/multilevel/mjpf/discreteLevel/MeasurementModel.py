from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreteLevel.State import State
from mDynamicSystem.state.measurement.Model import Model as MainMeasurementModel

class MeasurementModel(MainMeasurementModel):
    '''Should convert a superstate to a obs'''
    def __init__(self, superState:State):
        super().__init__(superState,None,None)

    def __getMeasurementWithoutNoise(self) -> Obs:
        return Obs(self._state.getRefVec())

