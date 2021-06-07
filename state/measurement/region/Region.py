from typing import List

from mDynamicSystem.measurement.Measurement import Measurement
import abc

class Region():
    '''To build a region based on measurements'''
    def __init__(self,measurements:List[Measurement]):
        self._measurements = measurements
        self._boundries = None

