from mDynamicSystem.state.State import State


class StateProbability:
    '''To represent posterior probabilities in intrest regions'''
    def __init__(self,state:State,probability:float):
        '''
        @todo no probability can be huger than 1
        @todo summation of all probabilities must be equal to 1
        :param state:
        :param probability:
        '''
        self.__state:State = state
        self.__probability = probability
