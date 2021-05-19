from mDynamicSystem.state.measurement.Model import Model as MainMeasurmentModel

class MeasurementModel(MainMeasurmentModel):
    def __init__(self, state):
        super().__init__(state)
