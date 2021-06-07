from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.decorator.Decorator import Decorator


class StateMorph(Decorator):
    def getMeasurementWithoiutNoise(self) -> Measurement:
        '''

        :return:
        '''
        measurementVector = self._state.getRefVec()
        measurement = Measurement(measurementVector)
        return measurement