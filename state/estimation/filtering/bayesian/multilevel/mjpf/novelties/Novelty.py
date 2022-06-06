import random
import abc

from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from typing import List
import numpy as np

from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.NoveltyComputer import NoveltyComputer
from mMath.linearAlgebra.Vector import Vector


class Novelty(metaclass=abc.ABCMeta):
    def __init__(self):
        self._numberOfClusters = 1
        self._timeStepStart = 0
        self._timeStepEnd = 4000
        self._timeStepStep = 10
        self._timeStepValues = None

        self._noveltyValues = []
        self._normalNoveltyValues = []
        self._noveltyComputer = NoveltyComputer()


    @abc.abstractmethod
    def _getNormalNoveltyValueByTimeStep(timeStep)->float:
        pass

    @abc.abstractmethod
    def getNoveltyValues(self)->List[float]:
        pass


    def getTimeStepStart(self):
        return self._timeStepStart

    def getTimeStepEnd(self):
        return self._timeStepEnd

    def _getTimeStepValues(self):
        if self._timeStepValues is None:
            self._timeStepValues = []
            for x in range(self._timeStepStart, self.getTimeStepEnd(), self._timeStepStep):
                self._timeStepValues.append(x)
        return self._timeStepValues

    def _getSmoothed(self, timeStepValues:List[int], noveltyValues:List[int]):
        model = make_interp_spline(timeStepValues, noveltyValues)
        timeStepValues = np.linspace(min(timeStepValues), max(timeStepValues), max(timeStepValues) - min(timeStepValues))
        noveltyValues = model(timeStepValues)
        return [timeStepValues, noveltyValues]

    def _getNormlizedValues(self, values:List[float],coefficient:float=1):
        maxValue = max(values)
        minValue = min(values)
        for yValueCounter,value in enumerate(values):
            values[yValueCounter] = coefficient*(value - minValue) / (maxValue - minValue)
        return values

    def _getNormalNoveltyValues(self):
        normalNoveltyValues = []
        for timeStep in self._timeStepValues:
            normalNoveltyValues.append(self._getNormalNoveltyValueByTimeStep(timeStep))
        normalNoveltyValues = self._getNormlizedValues(normalNoveltyValues,self._normalNoveltyValueCoefficient)
        return normalNoveltyValues

    def getMeanPlusVarOfNormal(self):
        noveltyValues = self._getNormalNoveltyValues()
        std = np.std(noveltyValues)
        mean = np.mean(noveltyValues)
        return mean+0.1*std

    def getAbstractStateChangesTimeStep(self):
        # abstract time step changes
        timeStepsAtWhichAbstractClassesChange = []
        clusterTimeSteps = int((self._timeStepEnd-self._timeStepStart)/self._numberOfClusters)
        for i in range(self._timeStepStart, self._timeStepEnd, clusterTimeSteps):
            if i == 0:
                continue
            timeStepsAtWhichAbstractClassesChange.append(random.randint(i - 10, i + 20))
        return timeStepsAtWhichAbstractClassesChange

