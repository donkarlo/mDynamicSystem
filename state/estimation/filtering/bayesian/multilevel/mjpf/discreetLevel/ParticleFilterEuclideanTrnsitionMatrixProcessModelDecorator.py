from typing import List
from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.PolynomialDrawParticles import \
    PolynomialDrawParticles
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.MeasurementSerie import MeasurementSerie
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.discreetLevel.SuperState import SuperState
from mDynamicSystem.state.estimation.process import Model as ProcessModel
from mDynamicSystem.state.measurement.decorator.Decorator import Decorator as ParticleFileterDecorator
from mMath.data.cluster.gng.Cluster import Cluster
from mMath.data.cluster.gng.examples.trajectory.Trajectory import Trajectory
from mMath.data.probability.discrete.Population import Population as DiscreetPopulation
from mMath.data.timeSerie.stochasticProcess.state.Serie import Serie as StateSerie
from mMath.data.timeSerie.stochasticProcess.state.SerieBuilder import SerieBuilder
from mMath.data.timeSerie.stochasticProcess.state.StateSet import StateSet
from mMath.data.timeSerie.stochasticProcess.state.transitionMatrix.FromStateSerieBuilder import FromStateSerieBuilder
from mDynamicSystem.state.estimation.process.decorator.StateTransitionMatrix import StateTransitionMatrix as StateTransitionMatrixProcessModel
from mDynamicSystem.state.estimation.process.decorator.Concrete import Concrete as ConcreteProcessModel
from mMath.data.probability.discrete.uniform.Uniform import Uniform as DiscreetUniform
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.Conceret import Conceret as ConcreteParticleFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.GaussianLikelihoodByState import GaussianLikelihoodByState as GaussianLikelihoodByStateParticleFilter
from mDynamicSystem.state.estimation.filtering.bayesian.monteCarlo.particle.decorator.GaussianDrawPrticles import GaussianDrawParticles as GaussianDrawParticlesParticleFilter
from mMath.region.Discreet import Discreet as DiscreetRegion
from mDynamicSystem.state.measurement.Model import Model as MeasurementModel


class ParticleFilterEuclideanTrnsitionMatrixProcessModelDecorator(ParticleFileterDecorator):
    def __init__(self):
        self._measurementSerie = MeasurementSerie()
        self._clusters: List[Cluster] = []
        self._superStateSet:StateSet = None
        self._superStateSerie:StateSerie = None

        ''''''
        #Measurements
        offlineMeasurementSerie = self._measurementSerie.getOfflineMeasurementSerie()



        #Process Model
        processModel = self.getProcessModel()


        #Measurement Model
        measurementModel =self._getMeasurementModel()

        #particle
        particlesNum = 10
        intrestedRegion:DiscreetRegion = DiscreetRegion()
        discreetPopulation = DiscreetPopulation()

        regionSampler = DiscreetUniform(discreetPopulation)
        startingState = offlineMeasurementSerie[0]
        concreteParticleFilter = ConcreteParticleFilter(particlesNum
                                ,self._getSuperStateSerie()
                                ,regionSampler
                                ,startingState
                                ,processModel
                                ,measurementModel)
        particleFilter = GaussianDrawParticlesParticleFilter(GaussianLikelihoodByStateParticleFilter(concreteParticleFilter))
        super().__init__(particleFilter)

    def _getSuperStateSet(self)->StateSet:
        '''

        :return:
        '''
        loopingCluster:Cluster
        if self._superStateSet.getLength() == 0:
            for loopingCluster in self._getClusters():
                superState:SuperState = SuperState(loopingCluster.getVectors().getMeanVector(),loopingCluster.getVectors().getVarianceVector())
                self._superStateSet.addState(superState)
        return self._superStateSet

    def _getClusters(self)->List[Cluster] :
        '''

        :return:
        '''
        if len(self._superStateSet) == 0:
            trajectoryGng = Trajectory(200)
            self._clusters = trajectoryGng.getClusters()
        return self._clusters

    def _getMeasurementBelongingSuperState(self, measurement:Measurement):
        '''
        To which superstate this measurement belongs?
        :param measurement:
        :return:
        '''
        measurementRefVec = measurement.getRefVec()
        loopingSuperState: SuperState
        superStateFoundForMeasurement = False
        belongingSuperState = None
        for loopingSuperState in self._getSuperStateSet():
            if loopingSuperState.getCenter().getDistanceFrom(measurementRefVec) < 3 * loopingSuperState.getStandardDeviation().getNorm():
                belongingSuperState = loopingSuperState
                superStateFoundForMeasurement = True
        if superStateFoundForMeasurement == False:
            belongingSuperState = self._getSuperStateSet().getARandomMember()
        return belongingSuperState

    def _getSuperStateSerie(self):
        '''
        :return:
        '''
        if self._superStateSerie is None:
            stateSerieBuilder: SerieBuilder = SerieBuilder(self._getSuperStateSet())
            loopingOfflineMeasurement: Measurement
            for loopingOfflineMeasurement in self._measurementSerie.getOfflineMeasurementSerie():
                stateSerieBuilder.appendState(self._getMeasurementBelongingSuperState(loopingOfflineMeasurement))
            self._superStateSerie = stateSerieBuilder.getSerie()
        return self._superStateSerie

    def getProcessModel(self)->ProcessModel:
        '''
        :return:
        '''
        transitionMatrix = FromStateSerieBuilder(self._getSuperStateSerie())
        processModel = PolynomialDrawParticles(StateTransitionMatrixProcessModel(ConcreteProcessModel(), transitionMatrix))
        return processModel

    def _getMeasurementModel(self) -> MeasurementModel:
        return self._measurementModel


if __name__=="__main__":
    measurementSerie = MeasurementSerie()
    mjpfStylePf = ParticleFilterEuclideanTrnsitionMatrixProcessModelDecorator()
    for measurement in measurementSerie.getOnlineMeausrementSerie():
        mjpfStylePf.addMeasurement(measurement)
        #posteriors for all regions
        posteriors = mjpfStylePf.getPosteriors()
        #prints posteriors for all state space
        print(posteriors)
