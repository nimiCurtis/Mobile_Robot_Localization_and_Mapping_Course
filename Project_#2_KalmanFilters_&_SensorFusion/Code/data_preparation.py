import numpy as np
import copy
from car_movement import CarMovement


def build_LLA_GPS_trajectory(dataset):
    lla_gps = None
    for gps_imu in dataset.get_gps_imu():
        gps_imu = gps_imu[0]
        currect_lla = np.array([gps_imu.lon, gps_imu.lat])
        
        if lla_gps is None:
            lla_gps = currect_lla.copy()
            continue
        
        lla_gps = np.vstack((lla_gps, currect_lla))
    
    return lla_gps


def build_GPS_trajectory(dataset):
    """
    That function gets a kitti dataset, and returns locations, times, yaw vf and wz values

    Args:
        dataset (DataLoader): kitti dataset structure to build trajectory from

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): array of locations (ENU), array of times, array of yaw, vf and wz values
    """
    locations = None
    times = None
    yaw_vf_wz = None
    
    for idx, (gps_imu, timestamp) in enumerate(zip(dataset.get_gps_imu(), dataset.get_timestamps())):
        gps_imu = gps_imu[0]
        
        if idx == 0:
            car_movement = CarMovement(gps_imu, timestamp)
        
        curr_yaw_vf_wz = np.array([gps_imu.yaw, gps_imu.vf, gps_imu.wz])
        curr_time = car_movement.calc_time(timestamp)
        
        NED2ENU = lambda NED: NED.dot(np.array([[0, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, -1]]))
        xNorth_yEast_zDown = np.array(car_movement.calc_movement(gps_imu))
        
        #yEast_xNorth_zUp = NED2ENU(xNorth_yEast_zDown)
        yEast_xNorth_zUp = np.array(car_movement.calc_movement(gps_imu))

        if locations is None:
            yaw_vf_wz = np.array([curr_yaw_vf_wz])
            locations = np.array([yEast_xNorth_zUp])
            times = np.array([curr_time])
        else:
            yaw_vf_wz = np.vstack((yaw_vf_wz, curr_yaw_vf_wz))
            locations = np.vstack((locations, yEast_xNorth_zUp))
            times = np.append(times, curr_time)
    
    return locations, times, yaw_vf_wz

def add_gaussian_noise(samples, sigma):
    assert type(samples) is np.ndarray, "samples must be ndarray type"
    assert len(samples.shape) == 1, "samples must be 1D array"
    assert type(sigma) is float or type(sigma) is int, "sigma type must be float or int"
    
    samples_count = samples.shape[0]
    mu = 0
    
    noise = np.random.normal(mu, sigma, samples_count)
    noised_samples = samples + noise
    return noised_samples


def add_gaussian_noise_dict(samples_dict, sigma_arr):
    samples_count = int(len(list(samples_dict.values()))/2)
    mu = 0
    
    noise_dist0 = np.random.normal(mu, sigma_arr[0], samples_count)
    noise_dist1 = np.random.normal(mu, sigma_arr[1], samples_count)
    noise_dist2 = np.random.normal(mu, sigma_arr[2], samples_count)
    
    samples_dict_noised = copy.deepcopy(samples_dict)
    for idx, (noise0, noise1, noise2) in enumerate(zip(noise_dist0, noise_dist1, noise_dist2)):
        samples_dict_noised[(idx, "odometry")]["r1"] += noise0
        samples_dict_noised[(idx, "odometry")]["t"] += noise1
        samples_dict_noised[(idx, "odometry")]["r2"] += noise2
    
    return samples_dict_noised


def normalize_angle(angle):
    if -np.pi < angle <= np.pi:
        return angle
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle <= -np.pi:
        angle = angle + 2 * np.pi
    return normalize_angle(angle)


def normalize_angles_array(angles):
    z = np.zeros_like(angles)
    for i in range(angles.shape[0]):
        z[i] = normalize_angle(angles[i])
    return z