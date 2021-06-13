from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.decorator.Decorator import Decorator


class StateMorph(Decorator):
    def getMeasurementRefVecWithoutNoise(self) -> Measurement:
        '''

        :return:
        '''
        measurementVector = self._state.getRefVec()
        return measurementVector