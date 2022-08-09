from mDynamicSystem.obs.Obs import Obs
from mDynamicSystem.state.estimation.filter.particle.decorator.Decorator import Decorator
from mMath.probability.continous.gaussian.Gaussian import Gaussian


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