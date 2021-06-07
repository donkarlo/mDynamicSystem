import abc
from mDynamicSystem.control import InputsSerie
from mDynamicSystem.state.State import State
from mDynamicSystem.measurement import Serie


class Estimation(metaclass=abc.ABCMeta):
    '''
    - The state estimate is represented by a pdf that quantifies both the estimated state and the uncertainty associated with the estimated value.
    -
    '''
    def __init__(self
                 , currentState:State
                 , inputControlsSerie:InputsSerie
                 , measurementsSerie: Serie
                 ):
        '''
        :param currentState:State
        :param inputControlsSerie: sequence of known control inputs
        :param measurementsSerie:
        '''
        self.__currentState = currentState
        self.__inputControlsSerie = inputControlsSerie
        self._measurementSerie = measurementsSerie
    @abc.abstractmethod
    def getPosteriors(self)->State:
        '''Get posteriors on state space'''
        pass