from mDynamicSystem.state.estimation.filtering.Filter import Filter as MainFilter
from mDynamicSystem.state.estimation.filtering.bayesian.particle.Particle import Particle
from mMath.data.probability.Event import Event
from mMath.data.probability.discrete.Pdf import Pdf
from mDynamicSystem.state.State import State
from mDynamicSystem.state.observation.Observation import Observation


class Filter(MainFilter):
    def __init__(self,numberOfParticles:int):
        ''''''
        self.__numberOfParticles = numberOfParticles

    def _getParticleWeightByParticle(self, particle: Particle):
        ''''''
        pdf = Pdf()
        lastObservationEvent:Observation = Event(self.getObservationsSerie().getLastObservation())
        lastStateEvent:State = Event(self.getStatesSerie().getLastState())
        updatedWeight = particle.getWeight() * pdf.getValueByEvent(lastObservationEvent.conditionedOn(lastStateEvent))
        return updatedWeight




