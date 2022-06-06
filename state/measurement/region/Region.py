from typing import List

from mDynamicSystem.obs.Obs import Obs
import abc

class Region():
    '''To build a region based on measurements'''
    def __init__(self, measurements:List[Obs]):
        self._measurements = measurements
        self._boundries = None

