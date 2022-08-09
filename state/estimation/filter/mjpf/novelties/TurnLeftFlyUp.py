from mDynamicSystem.state.estimation.filter.mjpf.novelties.Novelty import Novelty
import abc

class TurnLeftFlyUp(Novelty,metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self._timeStepEnd = 5333
        self._timeStepStep = 14