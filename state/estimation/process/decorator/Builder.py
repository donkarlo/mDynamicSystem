from mDynamicSystem.state.estimation.process.Concrete import Concrete as ConcereteProcessModel
from mDynamicSystem.state.estimation.process.decorator.StateTransitionMatrix import StateTransitionMatrix
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian as GassianPdf
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.TransitionMatrix import TransitionMatrix
from mMath.linearAlgebra.Vector import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix


class Builder:
    def getWithGaussianNoiseAndStateTransitionMatrix(previousState,gaussianMean:Vector,gaussianCovariance:Matrix,transitionMatrix:TransitionMatrix):
        #Buid Gaussian noise process decorator
        gaussianNoisePdf = GassianPdf(gaussianMean,gaussianCovariance)
        concreteProcessModel = ConcereteProcessModel(previousState
                                                     ,None #When there is a transitionMatrix no control inpu is needed
                                                     ,gaussianNoisePdf)
        processModel = StateTransitionMatrix(concreteProcessModel(),transitionMatrix)
        # finally has to call StateTransitionMatrix.getNextStateWithoutNoise first
        processModel.getNextMostProbableState()
