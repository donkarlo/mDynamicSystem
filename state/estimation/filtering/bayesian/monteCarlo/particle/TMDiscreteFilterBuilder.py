from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.Filter import Filter as ParticleFilter
from mDynamicSystem.state.estimation.process.Concrete import Concrete as ConcretePM
from mDynamicSystem.state.measurement import Concrete as ConcreteMM
from mDynamicSystem.state.measurement.decorator.StateMorph import StateMorph
from mMath.data.timeSerie.stochasticProcess.state.DiscreteStateSpace import DiscreteStateSpace
from mMath.data.timeSerie.stochasticProcess.state.State import State
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.TransitionMatrix import TransitionMatrix
from mDynamicSystem.state.estimation.process.decorator.StateTransitionMatrix import StateTransitionMatrix

class TMDiscreteFilterBuilder(ParticleFilter):
    '''
    - T: Transition Matrix, M:MorphMeasurement
    '''
    def __init__(self
                 , stateSpace:DiscreteStateSpace
                 , particlesNum:int
                 , stateTransitionMatrix:TransitionMatrix
                 ,startingState:State
                 ):

        processModel:StateTransitionMatrix = StateTransitionMatrix(ConcretePM(),stateTransitionMatrix)
        measurementModel = StateMorph(ConcreteMM())
        super().__init__(particlesNum
                         ,stateSpace
                         ,processModel
                         ,measurementModel
                         ,startingState)


