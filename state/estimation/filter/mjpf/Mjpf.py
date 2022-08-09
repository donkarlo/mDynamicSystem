from mDynamicSystem.state.estimation.Filter import Filter as BayesFilter
from mDynamicSystem.state.estimation.filter.particle.EuclideanMeasurementLikelihoodPdf import \
    EuclideanMeasurementLikelihoodPdf
from mDynamicSystem.state.estimation.filter.particle.Filter import Filter as ParticleFilter
from mDynamicSystem.state.estimation.filter.mjpf.OnlineMeasurementSerie import \
    OnlineMeasurementSerie
from mDynamicSystem.state.estimation.filter.mjpf.discreteLevel.StateSpace import \
    StateSpace
from mDynamicSystem.state.estimation.filter.mjpf.discreteLevel.TransitionMatrix import \
    TransitionMatrix
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel
from mDynamicSystem.state.estimation.process.Concrete import Concrete as ConcreteProcessModel
from mDynamicSystem.state.estimation.process import Model as ProcessModel
from mDynamicSystem.state.estimation.process.decorator.StateTransitionMatrix import StateTransitionMatrix as StateTransitionMatrixProcessModel
from mDynamicSystem.state.measurement.Concrete import Concrete as ConcreteMeasurementModel
from mDynamicSystem.state.measurement.decorator.StateMorph import StateMorph


class Mjpf(BayesFilter):
    '''Is formed of a discrete level and a continous level'''
    def __init__(self,particleFilter:ParticleFilter):
        # online measurements
        self._measurementSerie: OnlineMeasurementSerie = OnlineMeasurementSerie()
        self._particleFilter = None


if __name__=="__main__":
    superStateSpace = StateSpace()
    onlineMeasurementSerie = OnlineMeasurementSerie()
    startingSuperState = onlineMeasurementSerie.getFirstMeasurement()

    #particleFilterProcessModel
    concereteParticleFilterProcessModel = ConcreteProcessModel(startingSuperState)
    particleFilterProcessModel: ProcessModel = StateTransitionMatrixProcessModel(concereteParticleFilterProcessModel,TransitionMatrix())

    #Measurement model
    concreteParticleFilterMeasurementModel = ConcreteMeasurementModel()
    particleFilterMeasurementModel: MeasurementModel = StateMorph(concreteParticleFilterMeasurementModel)


    #particlefilter obs state likelihood
    particlefilterMeasurementLikelihood = EuclideanMeasurementLikelihoodPdf(superStateSpace, particleFilterMeasurementModel)

    # Particle filter
    particleFilter = ParticleFilter(16
                                    ,superStateSpace
                                    ,particleFilterProcessModel
                                    ,particleFilterMeasurementModel
                                    ,particlefilterMeasurementLikelihood
                                    ,startingSuperState)

    for onlineMeasurement in onlineMeasurementSerie.getMeasurementList():
        particleFilter.appendMeasurement(onlineMeasurement)
        print(particleFilter.getMaximumPosteriorStateProbability().getState().getRefVec())






