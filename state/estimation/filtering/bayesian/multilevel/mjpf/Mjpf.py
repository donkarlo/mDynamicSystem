from typing import List
from mDynamicSystem.state.estimation.filtering.bayesian.Filter import Filter
from mDynamicSystem.state.measurement.decorator.StateMorph import StateMorph as StateMorphMeasurementModel
from mDynamicSystem.state.measurement.decorator.Concrete import Concrete as ConcereteMeasurementModel



class Mjpf(Filter):

    def __init__(self):
        self._measurementMode = StateMorphMeasurementModel(ConcereteMeasurementModel())





