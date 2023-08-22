import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
import math
import itertools
import pandas as pd
import sys
np.random.seed(19)

from PF import ParticlesFilter

def read_landmarks(filename):
    # Reads the world definition and returns a list of landmarks, our 'map'.
    # .
    # The returned dict contains a list of landmarks each with the
    # following information: {id, [x, y]}

    landmarks = []

    f = open(filename)

    for line in f:
        line_s = line.split('\n')
        line_spl = line_s[0].split(',')
        landmarks.append([float(line_spl[0]), float(line_spl[1])])

    return landmarks


def read_odometry(filename):
    # Reads the odometry and sensor readings from a file.
    #
    # The data is returned in a dict where the u_t and z_t are stored
    # together as follows:
    #
    # {odometry,sensor}
    #
    # where "odometry" has the fields r1, r2, t which contain the values of
    # the identically named motion model variables, and sensor is a list of
    # sensor readings with id, range, bearing as values.
    #
    # The odometry and sensor values are accessed as follows:
    # odometry_data = sensor_readings[timestep, 'odometry']
    # sensor_data = sensor_readings[timestep, 'sensor']

    sensor_readings = dict()

    first_time = True
    timestamp = 0
    f = open(filename)

    for line in f:

        line_s = line.split('\n')  # remove the new line character
        line_spl = line_s[0].split(' ')  # split the line

        if line_spl[0] == 'ODOMETRY':
            sensor_readings[timestamp] = {'r1': float(line_spl[1]), 't': float(line_spl[2]), 'r2': float(line_spl[3])}
            timestamp = timestamp + 1

    return sensor_readings

def calculate_true_traj(trueOdometry):
    trueTrajectory = np.zeros((trueOdometry.__len__(), 3))
    for i in range(1, trueOdometry.__len__()):
        dr1 = trueOdometry[i - 1]['r1']
        dt = trueOdometry[i - 1]['t']
        dr2 = trueOdometry[i - 1]['r2']
        theta = trueTrajectory[i - 1, 2]
        #TODO fill odometry model
        dMotion = np.expand_dims(np.array([dt*np.cos(theta + dr1), dt*np.sin(theta + dr2)   ,  dr1 + dr2]), 0)  
        trueTrajectory[i, :] = trueTrajectory[i-1, :] + dMotion
    return trueTrajectory

def generate_measurment(trueOdometry, sigma_r1,sigma_t,sigma_r2):
    measurmentOdometry = dict()
    measured_trajectory = np.zeros((trueOdometry.__len__() + 1, 3))
    for i, timestamp in enumerate(range(trueOdometry.__len__())):
        #TODO
        dr1 = trueOdometry[timestamp]['r1'] + float(np.random.normal( 0,  sigma_r1, 1))  #TODO fill gaussian noise parameter
        dt = trueOdometry[timestamp]['t'] + float(np.random.normal( 0,  sigma_t, 1))    #TODO fill gaussian noise parameter
        dr2 = trueOdometry[timestamp]['r2'] + float(np.random.normal( 0,  sigma_r2, 1))  #TODO fill gaussian noise parameter
        measurmentOdometry[timestamp] = {'r1': dr1,
                                        't': dt,
                                        'r2': dr2}
        theta = measured_trajectory[i, 2]
        #TODO
        dMotion = np.expand_dims(np.array([dt*np.cos(theta + dr1), dt*np.sin(theta + dr2)   ,  dr1 + dr2]), 0)
        measured_trajectory[i + 1, :] = measured_trajectory[i, :] + dMotion

    return measurmentOdometry, measured_trajectory

def draw_pf_frame(trueTrajectory, measured_trajectory, trueLandmarks, particles, title, ax):
    """
    Plots the ground truth and estimated trajectories as well as the landmarks, the particles and their heading
    Parameters:
        trueTrajectory - dim is [num_frames x 3] or [num_frames x 2] (the heading is not used)
        measured_trajectory - dim is [num_frames x 3] or [num_frames x 2] (the heading is not used)
        trueLandmarks - dim is [num_landmarks, 2]
        particles - dim is [number_of_particles, 3]
        title - the title of the graph
    """
    ax.cla()
    ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    line_segments = []
    for particle in particles:
        x = particle[0]
        y = particle[1]
        heading_line_len = 0.5
        endx = x + heading_line_len * np.cos(particle[2])
        endy = y + heading_line_len * np.sin(particle[2])
        line_segments.append(np.array([[x, y], [endx, endy]]))
    line_collection = LineCollection(line_segments, color='c', alpha=0.08)
    ax.scatter(particles[:, 0], particles[:, 1], s=8, facecolors='none', edgecolors='g', alpha=0.7)
    ax.add_collection(line_collection)
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')

    ax.grid()
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.legend(['Ground Truth', 'Particle filter estimated trajectory', 'Landmarks', 'Particles and their heading'], prop={"size": 10}, loc="best")


def run(pf:ParticlesFilter, trueLandmarks,trueTrajectory, trueOdometry):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, timestamp in enumerate(range(0,trueOdometry.__len__() - 1)):
        # Observation model 
        # calculate Zt - the range and bearing to the closest landmark as seen from the current true position of the robot
        # TODO (norma 1) hint (use, np.argmin,np.linalg.norm, trueLandmarks ,trueTrajectory[i + 1, 0:2])
        closest_landmark_id = np.argmin(np.linalg.norm(trueLandmarks-trueTrajectory[i+1 , 0:2],axis=1))
        dist_xy = trueLandmarks[closest_landmark_id] - trueTrajectory[i+1, 0:2]
        # TODO (norma 1)
        r = np.linalg.norm(dist_xy) 
        # TODO 
        phi = np.arctan2(dist_xy[1],dist_xy[0]) - trueTrajectory[i+1,2]
        # TODO (add noise)
        r += np.random.normal(0,pf.sigma_range)
        # TODO (add noise)
        phi += np.random.normal(0,pf.sigma_bearing)
        phi = ParticlesFilter.normalize_angle(phi)
        Zt = np.array([r, phi])

        pf.apply(Zt, trueOdometry[timestamp])
        if i % 10 == 0 and i!=0:
            title = "pf_estimation_frame:{}_{}_particles".format(i, pf.numberOfParticles)
            draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles, title.replace("_", " ").replace("pf", "Particle filter"), ax=ax)
            plt.pause(0.1)  # Pause to update the plot dynamically
        
    plt.show()

def calculate_mse(X_Y_GT, X_Y_est, start_frame=50):
    """
    calculate MSE

    Args:
        X_Y_GT (np.ndarray): ground truth values of x and y
        X_Y_est (np.ndarray): estimated values of x and y

    Returns:
        float: MSE
    """
    error_x = X_Y_GT[:,0] - X_Y_est[:,0]
    error_y = X_Y_GT[:,1] - X_Y_est[:,1]

    # Calculate the squared error between the ground truth and estimated values
    squared_error_x = error_x ** 2
    squared_error_y = error_y ** 2

    min_idx = start_frame
    # Calculate the mean of the squared error to get the RMSE
    MSE = np.mean(squared_error_x[min_idx:] + squared_error_y[min_idx:])
    return float(MSE)

def display_true_trajectory(trueTrajectory, trueLandmarks):
    plt.figure(figsize=(8, 8))
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    plt.grid()
    plt.axis('equal')
    plt.xlabel("X [m]", fontsize=10)
    plt.ylabel("Y [m]", fontsize=10)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 8}, loc="best")
    plt.title('Ground trues trajectory and landmarks', fontsize=10)
    plt.show()

def display_measured_trajectory(trueLandmarks, trueTrajectory, measured_trajectory):
    plt.figure(figsize=(8, 8))
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=10)
    plt.ylabel("Y [m]", fontsize=10)
    plt.legend(['Ground truth trajectory', 'Trajectory with gaussian noise in the odometry data', 'Landmarks'], prop={"size": 10}, loc="best")
    plt.title('Ground trues trajectory, landmarks and noisy trajectory', fontsize=10)
    plt.show()

    
def main():
    # calculate true trajectory
    trueLandmarks = np.array(read_landmarks("Landmarks/LastID_6.csv")) #TODO -load your map
    trueOdometry = read_odometry("odometry.dat")
    trueTrajectory = calculate_true_traj(trueOdometry)
    display_true_trajectory(trueTrajectory, trueLandmarks)


    # generate measurment
    sigma_r1 = 0.01
    sigma_t = 0.2
    sigma_r2 = 0.01

    measurmentOdometry, measured_trajectory = generate_measurment(trueOdometry,sigma_r1,sigma_t,sigma_r2)
    display_measured_trajectory(trueLandmarks, trueTrajectory, measured_trajectory)

    # create pf object
    sigma_range = 1 
    sigma_bearing = 0.1  
    num_particles= 1000
    
    pf = ParticlesFilter(trueLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfPaticles=num_particles)

    # run pf
    run(pf=pf,
        trueLandmarks=trueLandmarks,
        trueTrajectory=trueTrajectory,
        trueOdometry=trueOdometry)
    
    mse = round(calculate_mse(trueTrajectory, pf.history, start_frame=10),ndigits=3)
    print("Root Mean Square error for {} particles is : {}".format(num_particles,mse))



if __name__ == '__main__':
    main()