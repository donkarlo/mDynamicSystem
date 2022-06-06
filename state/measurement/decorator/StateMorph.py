from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.state.measurement.decorator.Decorator import Decorator


class StateMorph(Decorator):
    def getMeasurementRefVecWithoutNoise(self) -> Obs:
        '''

        :return:
        '''
        measurementVector = self._state.getRefVec()
        return measurementVector