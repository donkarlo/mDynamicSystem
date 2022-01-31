from mDynamicSystem.obs.Obs import Obs


class MeasurementProbability:
    '''To represent posterior probabilities in intrest regions'''
    def __init__(self, measurement:Obs, probability:float):
        '''
        @todo no probability can be huger than 1
        @todo summation of all probabilities must be equal to 1
        :param state:
        :param probability:
        '''
        self._measurement:Obs = measurement
        self._probability = probability
