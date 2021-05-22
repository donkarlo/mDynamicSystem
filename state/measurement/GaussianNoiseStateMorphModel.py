from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian as GaussianPdf


class GaussianNoiseStateMorphModel(MeasurementModel):
    def __init__(self,gaussianPdf:GaussianPdf):
        self._gaussianPdf=gaussianPdf

    def getGaussianPdf(self):
        return self._gaussianPdf

    def getMeasurement(self) -> Measurement:
        measurementVector = self._state.getRefVec()+self._gaussianPdf.getASample()
        measurement = Measurement(measurementVector)
        return measurement

    def getMeasurementWithoiutNoise(self)->Measurement:
        measurementVector = self._state.getRefVec()
        measurement = Measurement(measurementVector)
        return measurement