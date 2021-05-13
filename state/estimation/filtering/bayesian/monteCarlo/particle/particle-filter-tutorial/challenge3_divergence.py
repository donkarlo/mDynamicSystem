#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Load variables
from shared_simulation_settings import *

# Particle filters
from core.particle_filters import ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    This file demonstrates particle filter that is caused by setting a too low number of particles. Note that (due to 
    the stochastic nature of the filter it could happen that the filter works fine during this short simulation. 
    However, that will be the exception rather than the rule. 
    """

    print("Starting demonstration of particle filter divergence.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(world_size_x, world_size_y, landmark_positions)
    visualizer = Visualizer()
    num_time_steps = 20

    # Initialize simulated robot
    robot = Robot(x=world.x_max * 0.75,
                  y=world.y_max / 5.0,
                  theta=3.14 / 2.0,
                  std_forward=true_robot_motion_forward_std,
                  std_turn=true_robot_motion_turn_std,
                  std_meas_distance=true_robot_meas_noise_distance_std,
                  std_meas_angle=true_robot_meas_noise_angle_std)

    ##
    # Particle filter settings
    # The process and measurement model noise is not equal to true noise.
    ##

    # Demonstrate divergence -> set number of particles too low
    number_of_particles = 100  # instead of 500 or even 1000

    # Initialize particle filter

    # Set resampling algorithm used (where applicable)
    resampling_algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise,
        resampling_algorithm)
    particle_filter_sir.initialize_particles_uniform()

    # Start simulation
    for i in range(num_time_steps):
        # Simulate robot move, perform measurements and update particle filter
        robot.move(robot_setpoint_motion_forward,
                   robot_setpoint_motion_turn,
                   world)
        measurements = robot.measure(world)
        particle_filter_sir.update(robot_setpoint_motion_forward,
                                   robot_setpoint_motion_turn,
                                   measurements,
                                   world.landmarks)

        # Visualize particles after initialization (to avoid cluttered visualization)
        visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    # Draw all particle at last time step
    visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    plt.show()