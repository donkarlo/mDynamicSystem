import abc
from mDynamicSystem.state.estimation import MeasurementEquation
from mDynamicSystem.state.estimation import ProcessModel


class Estimation:
    ''''''
    def __init__(self, stateEquation: ProcessModel, measurementEquation:MeasurementEquation):
        ''''''
        self.__stateEquation = stateEquation
        self.__measurementEquation = measurementEquation

    @abc.abstractmethod
    def getPosterior(self):
        '''Calculate p(x_{1:k}|z_{1:k}) if it is a markov process then it will be p(x_{k}|z_{1:k})'''
        pass