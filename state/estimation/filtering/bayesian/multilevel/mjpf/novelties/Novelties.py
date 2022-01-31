import random

import matplotlib.pyplot as plt

from ctumrs.topics.Plot3dColsFromTextFile import PlotFromCtuMrsTopicTextFile
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.ClusterSettings import ClusterSettings
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.Settings import Settings
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeft import TurnLeft
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftDb1 import TurnLeftDb1
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftDb2 import TurnLeftDb2
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftFlyUpDb1 import \
    TurnLeftFlyUpDb1
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftFlyUpDb2 import \
    TurnLeftFlyUpDb2
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftFlyUpInnovation import \
    TurnLeftFlyUpInnovation
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.TurnLeftInnovation import TurnLeftInnovation
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.Novelty import Novelty
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.UTurnDb1 import UTurnDb1
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.UTurnDb2 import UTurnDb2
from mDynamicSystem.state.estimation.filtering.bayesian.multilevel.mjpf.novelties.UTurnInnovation import UTurnInnovation
from mMath.data.cluster.gng.examples.trajectory.Trajectory import Trajectory
from mMath.trajectory.gen.examples.RectangleParamShapePointGenerator import RectangleParamShapePointGenerator
from mMath.trajectory.gen.examples.RectangleTurnLeftParamShapePointGenerator import \
    RectangleTurnLeftParamShapePointGenerator


class Novelties:
    def __init__(self):
        self._settings = Settings()
        cluster100Settings = ClusterSettings(20, 1)
        self._settings.setClusterSettings(cluster100Settings)

        #normal trajectory
        # tspsgRect = RectangleParamShapePointGenerator(0.5)
        # tspsgRect.plot3DPoints()
        #GPS Normal
        # plot = PlotFromCtuMrsTopicTextFile(
        #     "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/rect-10-0-5/gps.txt",
        #     ["field.pose.pose.position.x", "field.pose.pose.position.y", "field.pose.pose.position.z"])
        # plot.plot()

        #Abnormal
        # tspsg = RectangleTurnLeftParamShapePointGenerator(0.5)
        # tspsg.plot3DPoints()

        #GPS abnormal
        # plot = PlotFromCtuMrsTopicTextFile(
        #     "/home/donkarlo/Dropbox/projs/research/data/self-aware-drones/ctumrs/rect-10-0-5/gps-turn-left.txt",
        #     ["field.pose.pose.position.x", "field.pose.pose.position.y", "field.pose.pose.position.z"])
        # plot.plot()

        #Clustering
        # te = Trajectory(40000)
        # te.plot()
        pass

    def plotTurnLeftNoveltySignals(self):
        # Scale the plot
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(3)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('DB1,2 & innovation')

        turnLeftDb1 = TurnLeftDb1()
        db1AvoidanceNoveltySignals = turnLeftDb1.getNoveltyValues()
        plt.plot(db1AvoidanceNoveltySignals[0]
                 , db1AvoidanceNoveltySignals[1]
                 , label='DB1'
                 , color='green'
                 , linewidth=0.5)
        plt.plot(
            [turnLeftDb1.getTimeStepStart(), turnLeftDb1.getTimeStepEnd()]
            , [0.25, 0.25]
            , '--'
            , label='Thershold'
            , color='green')

        # DB2
        turnLeftDb2 = TurnLeftDb2()
        db2AvoidanceNoveltySignals = turnLeftDb2.getNoveltyValues()
        plt.plot(db2AvoidanceNoveltySignals[0]
                 , db2AvoidanceNoveltySignals[1]
                 , label='DB2'
                 , color='red'
                 , linewidth=0.5)

        plt.plot(
            [turnLeftDb2.getTimeStepStart(), turnLeftDb2.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='red')
        # Innovation
        turnLeftInnovation = TurnLeftInnovation()
        innovationAvoidanceNoveltySignals = turnLeftInnovation.getNoveltyValues()
        plt.plot(innovationAvoidanceNoveltySignals[0]
                 , innovationAvoidanceNoveltySignals[1]
                 , label='Innovation'
                 , color='blue'
                 , linewidth=0.5)

        #Mean line
        plt.plot(
            [turnLeftInnovation.getTimeStepStart(), turnLeftInnovation.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='blue')

        # Plotting  timestamps at which abstract states change using db1
        abstractLevelChangesTimeSteps = turnLeftDb1.getAbstractStateChangesTimeStep()
        for abstractLevelChangesTimeSteps in abstractLevelChangesTimeSteps:
            plt.plot(
                [abstractLevelChangesTimeSteps, abstractLevelChangesTimeSteps]
                , [-0.05, 1]
                , '--'
                , linewidth=0.5
                , color='purple')

        # plt.xlim([0, 800])
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend()

        # Novelty signal
        plt.show()


    def plotTurnLeftFlyUpNoveltySignals(self,settings=None):
        # Scale the plot
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(3)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('DB1,2 & innovation')

        turnLeftFlyUpDb1 = TurnLeftFlyUpDb1()
        db1AvoidanceNoveltySignals = turnLeftFlyUpDb1.getNoveltyValues()
        plt.plot(db1AvoidanceNoveltySignals[0]
                 , db1AvoidanceNoveltySignals[1]
                 , label='DB1'
                 , color='green'
                 , linewidth=0.5)
        plt.plot(
            [turnLeftFlyUpDb1.getTimeStepStart(), turnLeftFlyUpDb1.getTimeStepEnd()]
            , [0.25, 0.25]
            , '--'
            , label='Thershold'
            , color='green')

        # DB2
        turnLeftFlyUpDb2 = TurnLeftFlyUpDb2()
        db2AvoidanceNoveltySignals = turnLeftFlyUpDb2.getNoveltyValues()
        plt.plot(db2AvoidanceNoveltySignals[0]
                 , db2AvoidanceNoveltySignals[1]
                 , label='DB2'
                 , color='red'
                 , linewidth=0.5)

        plt.plot(
            [turnLeftFlyUpDb2.getTimeStepStart(), turnLeftFlyUpDb2.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='red')
        #Innovation
        turnLeftFlyUpInnovation = TurnLeftFlyUpInnovation()
        innovationAvoidanceNoveltySignals = turnLeftFlyUpInnovation.getNoveltyValues()
        plt.plot(innovationAvoidanceNoveltySignals[0]
                 , innovationAvoidanceNoveltySignals[1]
                 , label='Innovation'
                 , color='blue'
                 , linewidth=0.5)

        plt.plot(
            [turnLeftFlyUpInnovation.getTimeStepStart(), turnLeftFlyUpInnovation.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='blue')

        # Plotting  timestamps at which abstract states change using db1
        abstractLevelChangesTimeSteps = turnLeftFlyUpDb1.getAbstractStateChangesTimeStep()
        for abstractLevelChangesTimeSteps in abstractLevelChangesTimeSteps:
            plt.plot(
                [abstractLevelChangesTimeSteps, abstractLevelChangesTimeSteps]
                , [-0.05, 1]
                , '--'
                , linewidth=0.5
                , color='purple')

        # plt.xlim([0, 800])
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend()

        # Novelty signal
        plt.show()

    def plotUturnNoveltySignals(self):
        # Scale the plot
        f = plt.figure()
        f.set_figwidth(12)
        f.set_figheight(3)

        # Label
        plt.xlabel('Timestep')
        plt.ylabel('DB1,2 & innovation')

        uTurnDb1 = UTurnDb1()
        db1UturnNoveltySignals = uTurnDb1.getNoveltyValues()
        plt.plot(db1UturnNoveltySignals[0]
                 , db1UturnNoveltySignals[1]
                 , label='DB1'
                 , color='green'
                 , linewidth=0.5)
        plt.plot(
            [uTurnDb1.getTimeStepStart(), uTurnDb1.getTimeStepEnd()]
            , [0.25, 0.25]
            , '--'
            , label='Thershold'
            , color='green')

        # DB2
        uTurnDb2 = UTurnDb2()
        db2AvoidanceNoveltySignals = uTurnDb2.getNoveltyValues()
        plt.plot(db2AvoidanceNoveltySignals[0]
                 , db2AvoidanceNoveltySignals[1]
                 , label='DB2'
                 , color='red'
                 , linewidth=0.5)

        plt.plot(
            [uTurnDb2.getTimeStepStart(), uTurnDb2.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='red')
        # Innovation
        uTurnInnovation = UTurnInnovation()
        innovationAvoidanceNoveltySignals = uTurnInnovation.getNoveltyValues()
        plt.plot(innovationAvoidanceNoveltySignals[0]
                 , innovationAvoidanceNoveltySignals[1]
                 , label='Innovation'
                 , color='blue'
                 , linewidth=0.5)

        plt.plot(
            [uTurnInnovation.getTimeStepStart(), uTurnInnovation.getTimeStepEnd()]
            , [0.32, 0.32]
            , '--'
            , label='Thershold'
            , color='blue')

        # Plotting  timestamps at which abstract states change using db1
        abstractLevelChangesTimeSteps = uTurnDb1.getAbstractStateChangesTimeStep()
        for abstractLevelChangesTimeSteps in abstractLevelChangesTimeSteps:
            plt.plot(
                [abstractLevelChangesTimeSteps, abstractLevelChangesTimeSteps]
                , [-0.05, 1]
                , '--'
                , linewidth=0.5
                , color='purple')

        # plt.xlim([0, 800])
        # To show xlabel
        plt.tight_layout()

        # To show the inner labels
        plt.legend()

        # Novelty signal
        plt.show()

    def plot(self):
        # self.plotTurnLeftNoveltySignals()
        self.plotTurnLeftFlyUpNoveltySignals()
        # self.plotUturnNoveltySignals()
if __name__ == '__main__':
    novelties = Novelties()
    novelties.plot()