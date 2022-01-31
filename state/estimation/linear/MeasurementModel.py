from mDynamicSystem.state.measurement.Model import Model
from mDynamicSystem.state.estimation.filtering.bayesian.linear.kalman import Innovation
from mMath.linearAlgebra import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix
from mDynamicSystem.state import State
from mDynamicSystem.obs import Obs


class MeasurementModel(Model):
    '''z_k=Hx_{k}+v_{k}, v_{k}~N(0,R)

    '''

    def __init__(self
                 , measurementMatrix: Matrix
                 , measurementNoise: Matrix):
        self.__measurementMatrix = measurementMatrix
        self.__measurementNoise = measurementNoise

    def getMeasurementByState(self
                              , state: State
                              , onservationNoise: Vector = None) -> Measurement:
        pass

    def getInnovationObject(self
                            , observation: Measurement
                            , state: State) -> Innovation:
        """To calculate inoovation

        Parameters
        ----------
        observation:Obs
            a vector of obs
        state: State
            Prior Estimated State

        Returns
        -------
        invVec
        """
        inv = Innovation(self.__measurementMatrix, observation, state)
        return inv

    def getInnovation(self, zK: Measurement, xKPrEst: State) -> Vector:
        return self.getInnovationObject().getInnovation()

    def getObservationMatrix(self) -> Matrix:
        return self.__measurementMatrix

    def getObservationNoiseCov(self) -> Matrix:
        return self.__measurementNoise
