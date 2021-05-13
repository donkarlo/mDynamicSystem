from mDynamicSystem.state.estimation.MeasurementModel import MeasurementModel as MainMeasurmentModel

class MeasurementModel(MainMeasurmentModel):
    def __init__(self,currentState):
        super().__init__(currentState)
