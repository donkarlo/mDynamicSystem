from mDynamicSystem.state.measurement.Model import Model
from mDynamicSystem.state.measurement.ConcereteModel import ConcereteModel
from mDynamicSystem.state.measurement.GaussianNoiseDecorator import GaussianNoiseDecorator
from mDynamicSystem.state.measurement.StateMorphDecorator import StateMorphDecorator
from mMath.data.probability.continous.Gaussian import Gaussian
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.linearAlgebra.Vector import Vector
from mMath.linearAlgebra.matrix.Matrix import Matrix


class Builder:
    def getStateMorphWithGaussianNoise(self):
        '''
        :return:
        '''
        gaussianPdf = Gaussian(Vector([1,2])
                               ,Matrix([[0.1,0.2],[0.12,0.14]]))

        currentState:State = State(Vector([0,0]))
        controlInput:Vector = Vector([])
        measurementModel:Model = StateMorphDecorator(GaussianNoiseDecorator(gaussianPdf,ConcereteModel(currentState,controlInput)))
        return measurementModel