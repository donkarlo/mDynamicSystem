from mDynamicSystem.measurement.Measurement import Measurement


class MeasurementProbability:
    '''To represent posterior probabilities in intrest regions'''
    def __init__(self,measurement:Measurement,probability:float):
        '''
        @todo no probability can be huger than 1
        @todo summation of all probabilities must be equal to 1
        :param state:
        :param probability:
        '''
        self._measurement:Measurement = measurement
        self._probability = probability
