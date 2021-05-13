import abc
from mDynamicSystem.control import InputsSerie
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement import MeasurementsSerie


class Estimation(metaclass=abc.ABCMeta):
    ''''''
    def __init__(self
                 , currentState:State
                 , inputControlsSerie:InputsSerie
                 , measurementsSerie: MeasurementsSerie
                 ):
        '''
        :param currentState:State
        :param inputControlsSerie: sequence of known control inputs
        :param measurementsSerie:
        '''
        self.__currentState = currentState
        self.__inputControlsSerie = inputControlsSerie
        self.__measurementsSerie = measurementsSerie

    @abc.abstractmethod
    def getPosterior(self)->float:
        '''
        This is the result we expect to recieve from any estimation
        Calculate p(x_{1:k}|z_{1:k}) if it is a markov process then it will be p(x_{k}|z_{1:k})
        :return:
        '''
        pass