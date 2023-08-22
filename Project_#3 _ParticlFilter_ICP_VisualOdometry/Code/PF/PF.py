import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import os
import math
import itertools
import pandas as pd
import sys
import copy
import math
sys.path.append('/content')
np.random.seed(19)


class ParticlesFilter:
    def __init__(self, worldLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfPaticles=500):
        """
        Initialization of the particle filter
        """

        # Initialize parameters
        self.numberOfParticles = numberOfPaticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2

        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

        # Initialize particles - x, y, heading, weight (uniform weight for initialization)
        #TODO
        self.particles = np.concatenate((np.random.normal(0,2, (self.numberOfParticles, 1)),
                                         np.random.normal(0,2, (self.numberOfParticles, 1)),
                                         ParticlesFilter.normalize_angles_array(np.random.normal(0.1, 0.1, (self.numberOfParticles, 1)))), axis=1)
        
        #TODO fill inital weights
        self.weights = np.zeros((self.numberOfParticles, 1))
        self.weights.fill(1 / self.numberOfParticles)  
        self.history = np.array((0, 0, 0.1)).reshape(1, 3)
        self.particles_history = np.expand_dims(self.particles.copy(), axis=0)

    def apply(self, Zt, Ut):
        """
        apply the particle filter on a single step in the sequence
        Parameters:
            Zt - the sensor measurement (range, bearing) as seen from the current position of the car
            Ut - the true odometry control command
        """

        # Motion model based on odometry
        self.motionModel(Ut)

        # Measurement prediction
        ParticlesLocation = self.MeasurementPrediction()

        # Sensor correction
        self.weightParticles(Zt, ParticlesLocation)

        self.history = np.concatenate((self.history, self.bestKParticles(1).reshape(1, 3)), axis=0)

        # Resample particles
        self.resampleParticles()

        self.particles_history = np.concatenate((self.particles_history, np.expand_dims(self.particles.copy(), axis=0)), axis=0)

    def motionModel(self, odometry):
        """
        Apply the odometry motion model to the particles
        odometry - the true odometry control command
        the particles will be updated with the true odometry control command
        in addition, each particle will separately be added with Gaussian noise to its movement
        """
        #TODO (hint- use the input odometry + noise)
        dr1 = odometry['r1'] + np.random.normal( 0,  self.sigma_r1, (self.numberOfParticles, 1)) 
        dt = odometry['t'] + np.random.normal( 0,  self.sigma_t, (self.numberOfParticles, 1))   
        dr2 = odometry['r2'] + np.random.normal( 0,  self.sigma_r2, (self.numberOfParticles, 1)) 

        theta = self.particles[:, 2].reshape(-1, 1)
        #TODO fill odometer model
        dMotion = np.concatenate((
            dt*np.cos(theta + dr1),
            dt*np.sin(theta + dr1),
            dr1 + dr2), axis=1)

        self.particles = self.particles + dMotion
        self.particles[:, 2] = ParticlesFilter.normalize_angles_array(self.particles[:, 2])

    def MeasurementPrediction(self):
        """
        Calculates the measurement Prediction from the perspective of each of the particles
        returns: an array of size (number of particles x 2)
                 the first value is the range to the closest landmark and the second value is the bearing to it in radians
          
        """
        MeasurementPrediction = np.zeros((self.particles.shape[0], 2))  # range and bearing for each
        for i, particle in enumerate(self.particles):
            #TODO (hint- distance , calculate the closet Landmark location from each particle)
            closest_landmark_id  = np.argmin((np.linalg.norm(self.worldLandmarks-particle[:2],axis=1)))
            
            #TODO
            dist_xy = self.worldLandmarks[closest_landmark_id]-particle[:2]
            r = np.linalg.norm(dist_xy) 
            #TODO (hint-differecne between the theta (landmark--particle) minus the heading of the particle)
            phi = np.arctan2(dist_xy[1],dist_xy[0]) - particle[2]
            phi = ParticlesFilter.normalize_angle(phi)
 
            #TODO
            MeasurementPrediction[i, 0] = r 
            MeasurementPrediction[i, 1] = phi
        return MeasurementPrediction

    def weightParticles(self, car_measurement,MeasurementPrediction):
        """
        Update the particle weights according to the normal Mahalanobis distance
        Parameters:
            car_measurement - the sensor measurement to the closet landmark (range, bearing) as seen from the position of the car
            MeasurementPredction - the Particles locations (range, bearing) related to the landmark
        """
        # TODO ( sensor measurements covariance matrix)
        cov =  np.array([[self.sigma_range**2,0],
                        [0,self.sigma_bearing**2]])
        inv_cov = np.linalg.inv(cov)
        for i, relatedLocations in enumerate(MeasurementPrediction):
            d = car_measurement - relatedLocations
            d[1] = ParticlesFilter.normalize_angle(d[1])

            # TODO( hint: see normal distruntion , Multivariate Gaussian distributions)
            self.weights[i] = multivariate_normal.pdf(car_measurement, mean=relatedLocations, cov=cov)

        self.weights += 1.0e-200  # for numerical stability
        self.weights /= sum(self.weights)

    def resampleParticles(self):
        """
        law variance resampling
        """
        # TODO 

        self.particles = self.low_variance_sampler(self.particles,self.weights)
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def low_variance_sampler(self,particles, weights):
        n = len(particles)
        resampled_particles = np.zeros_like(particles)
        r = np.random.uniform(0, 1/n)
        c = weights[0]
        i = 0

        for m in range(n):
            U = r + m/n
            while U > c:
                i += 1
                c += weights[i]
            resampled_particles[m] = particles[i]

        return resampled_particles

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle to the range [-pi, pi]
        """
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle

    @staticmethod
    def normalize_angles_array(angles):
        """
        applies normalize_angle on an array of angles
        """
        z = np.zeros_like(angles)
        for i in range(angles.shape[0]):
            z[i] = ParticlesFilter.normalize_angle(angles[i])
        return z

    def bestKParticles(self, K):
        """
        Given the particles and their weights, choose the top K particles according to the weights and return them
        """
        indexes = np.argsort(-self.weights,axis=0)  # Sort the indices in descending order based on weights
        bestK = indexes[:K]  # Get the indices of the top K particles
        return self.particles[bestK, :]
    
