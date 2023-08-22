"""
KF Calibration

This script contains the classes and functions for performing calibration of Kalman Filter (KF) and Extended Kalman Filter (EKF) models. It includes the KfCalib class for KF calibration and the EkfCalib class for EKF calibration. The script also imports the KalmanFilter and ExtendedKalmanFilter classes from the kalman_filter module, as well as the graphs module for plotting and visualizing the calibration results.

Author: Nimrod Curtis
"""

import os
import numpy as np
import json
from hyperopt import fmin, tpe, atpe, hp
from kalman_filter import KalmanFilter, ExtendedKalmanFilter
import graphs

class KfCalib:
    """
    KF Calibration Class

    This class performs calibration of a Kalman Filter (KF) model. It takes a KF object, calibration folder path, and maximum number of iterations as inputs. The class provides methods to run the calibration and save the calibration logs.

    Args:
        kf (KalmanFilter): Kalman Filter object
        calib_folder (str): Path to the calibration folder
        max_iter (int): Maximum number of iterations for calibration (default: 10)
    """

    def __init__(self, kf: KalmanFilter, calib_folder, max_iter=10):
        self.max_iter = max_iter
        self.error_logs = {}
        self.iter = 0
        self.kf = kf
        self.calib_folder_path = calib_folder
        self.figs_dir = os.path.join(self.calib_folder_path, 'figs')
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)

    def _calibration_objective(self, params):
        """
        Calibration Objective Function

        This private method is used as the objective function for hyperparameter optimization during calibration. It initializes the Kalman Filter with the given parameters, runs the filter, calculates the RMSE and maxE errors, and saves the error logs.

        Args:
            params (dict): Dictionary containing the parameter values

        Returns:
            float: Root Mean Squared Error (RMSE)
        """
        # Initialize Kalman filter with given parameters
        self.kf.__setattr__('k', params['k'])
        self.kf.__setattr__('sigma_n', params['sigma_n'])
        
        # Run Kalman filter and obtain locations and covariance
        locations_kf, covariance_kf = self.kf.run()
        
        # Calculate RMSE and maxE errors
        rmse, maxE, error_x, error_y = self.kf.calc_RMSE_maxE(self.kf.enu_gt[:, :2], locations_kf[:, [0, 2]])
        
        x_cov, y_cov = covariance_kf[:, 0, 0], covariance_kf[:, 2, 2]
        graphs.plot_error((error_x, np.sqrt(x_cov)), (error_y, np.sqrt(y_cov)))
        graphs.show_graphs(folder=self.figs_dir, file_name=f"kf_error_k{np.round(params['k'], decimals=2)}_sigman{np.round(params['sigma_n'], decimals=2)}")

        # Save RMSE error to internal list
        self.error_logs[self.iter] = {'k': np.round(params['k'], decimals=2), 'sigma_n': np.round(params['sigma_n'], decimals=2), 'maxE': maxE, 'rmse': rmse}
        self.iter += 1

        return rmse
    
    def run_calibration(self):
        """
        Run Calibration

        This method performs the calibration by running the hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) algorithm. It prints the best hyperparameters found, saves the plots and parameter logs, and returns the best parameters.

        Returns:
            dict: Best hyperparameters found
        """
        print("Calibration start!")
        # Define parameter space for hyperparameter optimization
        parameter_space = {
            'k': hp.quniform('k', 1, 5, 0.2),
            'sigma_n': hp.quniform('sigma_n', 0, 5, 0.2)
        }
        
        # Perform hyperparameter optimization using TPE algorithm
        best_params = fmin(fn=self._calibration_objective,
                           space=parameter_space,
                           algo=tpe.suggest,
                           max_evals=self.max_iter)
        
        # Print best hyperparameters found
        print("Calibration finish!")
        print('Best hyperparameters found:', best_params)
        print(f'Save plots and params logs in: {self.calib_folder_path}')
        
        file_name = os.path.join(self.calib_folder_path, 'calib_logs.json')
        self.save_error_logs(filename=file_name)

        return best_params

    def save_error_logs(self, filename):
        """
        Save Error Logs

        This method saves the error logs to a JSON file.

        Args:
            filename (str): Name of the JSON file to save
        """
        # Save maxE errors to JSON file
        with open(filename, 'w') as f:
            json.dump(self.error_logs, f, indent=4)
    

class EkfCalib:
    """
    EKF Calibration Class

    This class performs calibration of an Extended Kalman Filter (EKF) model. It takes an EKF object, calibration folder path, maximum number of iterations, and stop loss threshold as inputs. The class provides methods to run the calibration and save the calibration logs.

    Args:
        ekf (ExtendedKalmanFilter): Extended Kalman Filter object
        calib_folder (str): Path to the calibration folder
        max_iter (int): Maximum number of iterations for calibration (default: 10)
        stop_loss (int): Stop loss threshold for early stopping (default: 3)
    """

    def __init__(self, ekf: ExtendedKalmanFilter, calib_folder, max_iter=10, stop_loss=3):
        self.max_iter = max_iter
        self.stop_loss = stop_loss
        self.error_logs = {}
        self.iter = 0
        self.ekf = ekf
        self.calib_folder_path = calib_folder
        self.figs_dir = os.path.join(self.calib_folder_path, 'figs')
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)

    def _calibration_objective(self, params):
        """
        Calibration Objective Function

        This private method is used as the objective function for hyperparameter optimization during calibration. It initializes the Extended Kalman Filter with the given parameters, runs the filter, calculates the RMSE and maxE errors, and saves the error logs.

        Args:
            params (dict): Dictionary containing the parameter values

        Returns:
            float: Root Mean Squared Error (RMSE)
        """
        # Initialize Kalman filter with given parameters
        self.ekf.__setattr__('k', params['k'])
        self.ekf.__setattr__('sigma_theta', params['sigma_theta'])
        self.ekf.__setattr__('sigma_r', [params['sigma_rxy'], params['sigma_rtheta']])

        # Run Kalman filter and obtain locations and covariance
        locations_ekf, covariance_ekf = self.ekf.run()
        x_y_theta_gt = np.concatenate([self.ekf.enu_gt[:, :2], self.ekf.yaw_vf_wz[:, 0][:, np.newaxis]], axis=1)
        # Calculate RMSE and maxE errors
        rmse, maxE, error_x, error_y, error_yaw = self.ekf.calc_RMSE_maxE(x_y_theta_gt, locations_ekf)

        x_cov, y_cov, yaw_cov = covariance_ekf[:, 0, 0], covariance_ekf[:, 1, 1], covariance_ekf[:, 2, 2]
        graphs.plot_error((error_x, np.sqrt(x_cov)), (error_y, np.sqrt(y_cov)), (error_yaw, np.sqrt(yaw_cov)))
        graphs.show_graphs(folder=self.figs_dir, file_name=f"ekf_error_k{np.round(params['k'], decimals=2)}_sigmatheta{np.round(params['sigma_theta'], decimals=2)}")

        # Save RMSE error to internal list
        self.error_logs[self.iter] = {'k': np.round(params['k'], decimals=2), 'sigma_theta': np.round(params['sigma_theta'], decimals=2), 'sigma_rxy': np.round(params['sigma_rxy'], decimals=2), 'sigma_rtheta': np.round(params['sigma_rtheta'], decimals=2), 'maxE': maxE, 'rmse': rmse}
        self.iter += 1
        return rmse

    def run_calibration(self):
        """
        Run Calibration

        This method performs the calibration by running the hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) algorithm. It prints the best hyperparameters found, saves the plots and parameter logs, and returns the best parameters.

        Returns:
            dict: Best hyperparameters found
        """
        print("Calibration start!")
        # Define parameter space for hyperparameter optimization
        parameter_space = {
            'k': hp.quniform('k', 1, 5, 0.2),
            'sigma_theta': hp.quniform('sigma_theta', 0.1, 5, 0.1),
            'sigma_rxy': hp.quniform('sigma_rxy', 0.05, 1, 0.05),
            'sigma_rtheta': hp.quniform('sigma_rtheta', 0.01, 0.5, 0.05)
        }
        
        # Perform hyperparameter optimization using Tree-structured Parzen Estimator (TPE) algorithm
        best_params = fmin(fn=self._calibration_objective,
                           space=parameter_space,
                           algo=atpe.suggest,
                           max_evals=self.max_iter)

                        #    early_stop_fn=lambda x: x['loss'] < self.stop_loss)  # stop if loss is less than threshold
        
        # Print best hyperparameters found
        print("Calibration finish!")
        print('Best hyperparameters found:', best_params)
        print(f'Save plots and params logs in: {self.calib_folder_path}')
        
        file_name = os.path.join(self.calib_folder_path, 'calib_logs.json')
        self.save_error_logs(filename=file_name)

        return best_params

    def save_error_logs(self, filename):
        """
        Save Error Logs

        This method saves the error logs to a JSON file.

        Args:
            filename (str): Name of the JSON file to save
        """
        # Save maxE errors to JSON file
        with open(filename, 'w') as f:
            json.dump(self.error_logs, f, indent=4)
