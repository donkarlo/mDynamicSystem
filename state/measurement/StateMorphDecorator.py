from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.Decorator import Decorator


class StateMorphDecorator(Decorator):

    def getMeasurement(self) -> Measurement:
        ''''''
        return self._model.getState() + self._model.getMeasurement()