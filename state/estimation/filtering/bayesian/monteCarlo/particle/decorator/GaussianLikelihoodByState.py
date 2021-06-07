from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.Decorator import Decorator
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian


class GaussianLikelihoodByState(Decorator):
    def _getMeasurementLikelihoodByState(self, state: State) -> float:
        '''
        :param state:
        :return:
        '''
        gaussian: Gaussian = Gaussian(self._measurementSerie.getLastMeasurement()
                                      , self._gaussianNoiseStateMorphMeasurementModel.getGaussianPdf().getCovariance())
        likelihood = gaussian.getValueByEvent(
            self._gaussianNoiseStateMorphMeasurementModel.getMeasurementWithoutNoise(state))
        return likelihood