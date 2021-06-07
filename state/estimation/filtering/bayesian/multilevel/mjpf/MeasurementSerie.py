from mDynamicSystem.state.measurement.Model import Model as MeasurementModel

from mMath.data.cluster.gng.examples.trajectory.ThreeDPosVelFile import ThreeDPosVelFile
from mDynamicSystem.measurement.Serie import Serie as MainMeasurementSerie
from mDynamicSystem.measurement.Measurement import Measurement
from mMath.linearAlgebra.Vector import Vector
class MeasurementSerie(MeasurementModel):


    def getOfflineMeasurementSerie(self)->MainMeasurementSerie:
        '''works as training data'''
        fileDataBank = "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/two-step/manip/pos-vel-measurement-from-gps.txt"
        t3dposVel = ThreeDPosVelFile(fileDataBank)
        offlineMeasurementSerie: MainMeasurementSerie
        t3posVelsNpArrays = t3dposVel.getNpArr(5000)
        for offlineMeasurementNpArray in t3posVelsNpArrays:
            offlineMeasurementSerie.appendMeasurement(Measurement(Vector(offlineMeasurementNpArray)))
        return offlineMeasurementSerie

    def getOnlineMeausrementSerie(self)->MainMeasurementSerie:
        '''works as testing data'''
        onlineMeasurementSerie:MainMeasurementSerie
        return onlineMeasurementSerie
