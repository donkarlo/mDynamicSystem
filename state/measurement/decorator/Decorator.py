from mDynamicSystem.state.measurement.Model import Model as MeasurementModel


class Decorator(MeasurementModel):
    def __init__(self, model:MeasurementModel):
        self._model = model
