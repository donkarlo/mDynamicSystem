from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.state.State import State
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.Decorator import Decorator
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian


class GaussianMeasurementStateLikelihood(Decorator):
    def _getMeasurementLikelihood(self, expectedMeasurement: Obs) -> float:
        '''
        :param state:
        :return:
        '''
        gaussian: Gaussian = Gaussian(self._measurementSerie.getLastMeasurement()
                                      , self._measurementModel.getNoisePdf().getCovariance())
        likelihood = gaussian.getValueAt(expectedMeasurement)
        return likelihood