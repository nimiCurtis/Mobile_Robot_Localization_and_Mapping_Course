"""
This script provides functions to analyze and plot calibration results obtained from the KfCalib and EkfCalib classes.
It includes functions to visualize calibration data using scatter plots and 3D heatmaps.

"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DIR = os.path.dirname(__file__)
RES_DIR = os.path.join(DIR, "../../Results")


def plot_kf_calib_analyze(logs_dic):
    """
    Plots the analysis of Kalman filter calibration results.

    Parameters:
    - logs_dic (dict): Dictionary containing calibration logs

    """
    data_dict = logs_dic
    # Create an array to store the values
    num_keys = len(data_dict)
    data_array = np.zeros((num_keys, 3))

    # Fill the array with the values from the dictionary
    for i, key in enumerate(data_dict):
        data = data_dict[key]
        data_array[i] = [data["k"], data["sigma_n"], data["rmse"]]

    # Extract k and sigma_n values for plotting
    k_values = data_array[:, 0]
    sigma_n_values = data_array[:, 1]

    # Create a scatter plot with color based on sigma_n
    plt.scatter(k_values, sigma_n_values, c=sigma_n_values, cmap='jet')
    plt.colorbar(label='rmse')

    # Set axis labels
    plt.xlabel('k')
    plt.ylabel('sigma_n')

    # Display the plot
    plt.show()


def plot_ekf_calib_analyze(logs_dic):
    """
    Plots the analysis of Extended Kalman filter calibration results.

    Parameters:
    - logs_dic (dict): Dictionary containing calibration logs

    """
    data_dict = logs_dic
    # Create an array to store the values
    num_keys = len(data_dict)
    data_array = np.zeros((num_keys, 4))

    # Fill the array with the values from the dictionary
    for i, key in enumerate(data_dict):
        data = data_dict[key]
        data_array[i] = [data["k"], data["sigma_theta"], data["sigma_rxy"], data["rmse"]]

    # Extract k, sigma_theta, sigma_rxy, and rmse values for plotting
    k_values = data_array[:, 0]
    sigma_theta_values = data_array[:, 1]
    sigma_rxy_values = data_array[:, 2]
    rmse_values = data_array[:, 3]

    # Create a 3D scatter plot with colors based on rmse
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(k_values, sigma_theta_values, sigma_rxy_values, c=rmse_values, cmap='jet')

    # Set the axis labels
    ax.set_xlabel('k')
    ax.set_ylabel('sigma_theta')
    ax.set_zlabel('sigma_rxy')
    ax.set_title('3D Heatmap')

    # Add a color bar
    cbar = fig.colorbar(scatter, label='rmse')

    # Display the plot
    plt.show()


def save_calib_analyze_plot(folder_name):
    """
    Saves and displays the analysis plot based on the calibration folder name.

    Parameters:
    - folder_name (str): Name of the calibration folder ('kf_calib' or 'ekf_calib')

    """
    calib_dir = os.path.join(RES_DIR, folder_name)
    logs_file = os.path.join(calib_dir, 'calib_logs.json')

    with open(logs_file, 'r') as f:
        logs_dic = json.load(f)

    if folder_name == 'kf_calib':
        plot_kf_calib_analyze(logs_dic)
    elif folder_name == 'ekf_calib':
        plot_ekf_calib_analyze(logs_dic)


def main():
    """
    Main function to save and display the calibration analysis plots.
    Uncomment the desired plot to be generated.

    """
    # Save heat map of kf and ekf calibration.
    # Uncomment the desired plot

    save_calib_analyze_plot('kf_calib')
    # save_calib_analyze_plot('ekf_calib')


if __name__ == "__main__":
    main()
