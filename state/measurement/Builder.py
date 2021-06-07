from mDynamicSystem.state.State import State
from mDynamicSystem.state.measurement.decorator.Concrete import Concrete
from mDynamicSystem.state.measurement.decorator.StateMorph import StateMorph
from mMath.data.probability.continous.gaussian.Gaussian import Gaussian
from mMath.linearAlgebra.Vector import Vector


class Builder():
    def getStateMorphGaussianNoiseModel(self,state:State,ControlInput:Vector,gaussianNoise:Gaussian,timeStep:int=None):
        concreteMeasurementModel = Concrete(state,ControlInput,gaussianNoise,timeStep)
        return StateMorph(concreteMeasurementModel)