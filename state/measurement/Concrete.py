from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.linearAlgebra.Vector import Vector


class Concrete(MeasurementModel):
    def getMeasurementRefVecWithoutNoise(self)->Vector:
        return None
