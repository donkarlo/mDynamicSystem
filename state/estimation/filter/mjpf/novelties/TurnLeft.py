from mDynamicSystem.state.estimation.filter.mjpf.novelties.Novelty import Novelty


class TurnLeft(Novelty):
    def __init__(self):
        super().__init__()
        self._timeStepEnd = 4666
        self._timeStepStep = 12