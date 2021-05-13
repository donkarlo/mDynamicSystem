from mMath.data.timeSerie.stochasticProcess.markov.MarkovChain import MarkovChain
from mDynamicSystem.state import MeasurementEquation
from mDynamicSystem.state.estimation.linear import StateModel


class Estimation(MarkovChain):
    ''''''
    def __init__(self, stateEquation: StateModel, measurementequation:MeasurementEquation):
        ''''''
        self.__stateEquation = stateEquation
        self.__measurementequation = measurementequation