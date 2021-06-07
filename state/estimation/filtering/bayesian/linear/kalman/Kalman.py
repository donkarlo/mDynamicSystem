from linearalgebra import Matrix, Vector

import Innovation
from mDynamicSystem.state import Filter
from mDynamicSystem.state import ObsModel
from mDynamicSystem.state import ObsSerie
from mDynamicSystem.state import State
from mDynamicSystem.state.estimation.linear import ProcessModel


class Kalman(Filter):
    """
    process processModel: x_k=Fx_{k-1}+Bu_{k-1}+w_k, F:State transition matrix, B: Control-Input matrix
    Process noise vector: w_k ~ N(0.Q), Q: Process noise covariance matrix

    Paired with
    -----
    measurement processModel: z_k=Hx_{k-1}=v_{k}
    measurement noise vector: v_k ~ N(0,R), R: measurement noise cov matrix

    Notation
    ---------
    From here: -: predicted/prior estimation

    Prediction
    -------
    Predicted state estimate x^{^-}_{k}=Fx^{^+}_{k-1}+Bu_{k-1}, -:prior dist +:postrior dist
    P^{-}_{k}=FP^{+}_{k-1}F^{T}+Q, P:state error covariance

    update
    -----------
    Mesurement residual: y^{~}_{k} = z_k-Hx^{^-}_{k}
    Kalman gain: K_{k} = P^{-}_{k}H^{T}(R+HP^{-}_{k}H^{T})^{-1}
    Updated state estimate: x^{^+}_{k}=x^{^-}_{k}+K_ky^{~}
    Updated error estimate: P^{+}_{k} = (I-K_{k}H)P^{-}_{k}

    initial guesses
    ----------------
    x0Post: initial guess of state filtering, initial guess of state
    p0Post: initial, posterior state error covariance, initial guess of error cov
    1. To keep up with intial ignorance rule please send larg values
    2. The more x0Post is unceratin the larger poPost should be


    f:Process matrix, b:Control matrix, q:Process Noise Cov matrix, h:Observation matrix, r:Observation Noise Cov matrix
    self._r = r
    self._h = h
    self._q = q
    self._f = f
    self._b = b

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self
                 , observationSeri: ObsSerie
                 , processModel: ProcessModel
                 , observationModel: ObsModel
                 , initialEstimatedState: State
                 , initialStateErrorCov: Matrix) -> None:
        super().init(observationSeri)
        self.__processModel = processModel
        self.__observationModel = observationModel
        self.__initialStateErrorCov = initialStateErrorCov
        self.__initialEstimatedState = initialEstimatedState

    def _predict(self) -> None:
        """"""
        pass

    def _update(self) -> None:
        """"""
        pass

    def __getLastObservation(self) -> Vector:
        return self._observationSeri[len(self._observationSeri)]

    def __getInnovation(self) -> Matrix:
        """
        """
        ino = Innovation(self.__processModel.getProcessMatrix()
                         , self.getLastObservation()
                         , self.getPriorPrediction())
        return ino.getInnovation()

    def __getCurrentPriorEstimatedState(self, prevPostEstimatedState: State, prevControl) -> State:
        '''x^{^-}_{k}=Fx^{^+}_{k-1}+Bu_{k-1}'''
        predictedPreX = self.__processModel.getNextMostProbableState(prevPostEstimatedState, prevControl)
        return predictedPreX

    def __getCurrentPriorStateErrorCov(self, previousPosteriorStateErrorCov) -> Matrix:
        '''P^{-}_{k}=FP^{+}_{k-1}F^{T}+Q, P:state error covariance
        '''
        f = self.__processModel.getProcessMatrix()
        q = self.__processModel.getProcessNoiseCov()
        currentPriorStateErrorCov = f * previousPosteriorStateErrorCov * f ** ('T') + q
        return currentPriorStateErrorCov

    def __getKolmanGain(self
                        , currentPriorStateErrorCov: Matrix):
        '''Kalman gain: K_{k} = P^{-}_{k}H^{T}(R+HP^{-}_{k}H^{T})^{-1}'''
        h = self.__observationModel.getObservationMatrix()
        r = self.__observationModel.getObservationNoiseCov()
        return currentPriorStateErrorCov * h ** ('T') * (r + h * currentPriorStateErrorCov * h ** ('T')) ** (-1)

    def __getCurrentPosteriorEstimatedState(self
                                            , priorCurrentEstimatedState: State
                                            , currentPriorStateErrorCov: Matrix,
                                            innovation: Innovation) -> State:
        '''Updated state estimate: x^{^+}_{k}=x^{^-}_{k}+K_ky^{~}'''
        kGain = self.__getKolmanGain(currentPriorStateErrorCov)
        CurrentPosteriorEstimatedState = priorCurrentEstimatedState + kGain * innovation.getInnovation()
        return CurrentPosteriorEstimatedState

    def __getCurrentPriorEstimatedState(self
                                        , currentPriorStateErrorCov: Matrix
                                        , innovation: Innovation) -> Matrix:
        '''Updated error estimate: P^{+}_{k} = (I-K_{k}H)P^{-}_{k}

        todo
        ----
        determine the dimention of the identity matrix
        '''
        p = (Matrix.getIdentity(1) - self.__getKolmanGain(currentPriorStateErrorCov)
             * self.__processModel.getProcessMatrix() + innovation.getInnovation()) \
            * self.__getCurrentPriorStateErrorCov()
        return p
