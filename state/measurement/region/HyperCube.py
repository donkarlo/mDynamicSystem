from typing import List

from mDynamicSystem.measurement.Measurement import Measurement
from mDynamicSystem.state.measurement.region.Region import IntrestedRegion, Region


class HyperCube(Region):
    '''To build a region based on measurements'''
    def __init__(self,measurements:List[Measurement]):
        self._measurements = measurements

    def getBoundries(self):
        for measurement in self._measurements:
            measurement:Measurement
            dimsMinMax = []
            for componentIndex,component in enumerate(measurement):
                component = component[0]
                if dimsMinMax[componentIndex][0] is None:
                    dimsMinMax[componentIndex][0] = 0
                if dimsMinMax[componentIndex][1] is None:
                    dimsMinMax[componentIndex][1] = 0
                if component < dimsMinMax[componentIndex][0]:
                    dimsMinMax[componentIndex][0] = component
                elif component > dimsMinMax[componentIndex][1]:
                    dimsMinMax[componentIndex][1] = component
        return dimsMinMax