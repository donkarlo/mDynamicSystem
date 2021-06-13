from mDynamicSystem.measurement.Serie import Serie as MeasurementSerie
from mMath.data.cluster.gng.examples.trajectory.ThreeDPosVelFile import ThreeDPosVelFile
from mDynamicSystem.measurement.Measurement import Measurement
from mMath.linearAlgebra.Vector import Vector
class TrainingMeasurementSerie(MeasurementSerie):
    def __init__(self):
        '''works as training data'''
        super().__init__()
        fileDataBank = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-step/manip/pos-vel-measurement-from-gps.txt"
        t3dposVel = ThreeDPosVelFile(fileDataBank)
        t3posVelsNpArrays = t3dposVel.getNpArr(3500)
        for trainingMeasurementNpArray in t3posVelsNpArrays:
            self.appendMeasurement(Measurement(Vector(trainingMeasurementNpArray)))
