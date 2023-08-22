import os
import numpy as np
from data_preparation import *
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation
import graphs
import random

from data_loader import DataLoader
from kalman_filter import KalmanFilter , ExtendedKalmanFilter , ExtendedKalmanFilterSLAM
from calibration import KfCalib, EkfCalib
import graphs
import random
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation
from data_preparation import build_GPS_trajectory, add_gaussian_noise

random.seed(42)
np.random.seed(42)

font = {'size': 20}
plt.rc('font', **font)


class ProjectQuestions:
    def __init__(self, dataset:DataLoader):
        """
        Given a Loaded Kitti data set with the following ground truth values: tti dataset and adds noise to GT-gps values
        - lat: latitude [deg]
        - lon: longitude [deg]
        - yaw: heading [rad]
        - vf: forward velocity parallel to earth-surface [m/s]
        - wz: angular rate around z axis [rad/s]
        Builds the following np arrays:
        - enu - lla converted to enu data
        - times - for each frame, how much time has elapsed from the previous frame
        - yaw_vf_wz - yaw, forward velocity and angular change rate
        - enu_noise - enu with Gaussian noise (sigma_xy=3 meters)
        - yaw_vf_wz_noise - yaw_vf_wz with Gaussian noise in vf (sigma 2.0) and wz (sigma 0.2)
        """
        self.dataset = dataset
        
        self.results_dir = os.path.join(os.path.dirname(__file__),'../../Results')
        self.figs_dir = os.path.join(self.results_dir,'figs')
        if not os.path.exists(self.figs_dir):
            os.mkdir(self.figs_dir)
        self.ani_dir = os.path.join(self.results_dir,'animations')
        if not os.path.exists(self.ani_dir):
            os.mkdir(self.ani_dir)
        self.kf_calib_dir = os.path.join(self.results_dir,'kf_calib')
        if not os.path.exists(self.kf_calib_dir):
            os.mkdir(self.kf_calib_dir)
        self.ekf_calib_dir = os.path.join(self.results_dir,'ekf_calib')
        if not os.path.exists(self.ekf_calib_dir):
            os.mkdir(self.ekf_calib_dir)

        #TODO (hint- use build_GPS_trajectory)
        self.enu, self.times, self.yaw_vf_wz = build_GPS_trajectory(self.dataset) 

        #TODO
        # add noise to the trajectory
        self.sigma_xy = 3 # [meters]
        self.sigma_vf = 2 # [meters/sec]
        self.sigma_wz = 0.2 # [rad/sec]

        #TODO
        self.enu_noise = np.column_stack((add_gaussian_noise(self.enu[:,0],self.sigma_xy),
                                            add_gaussian_noise(self.enu[:,1],self.sigma_xy),
                                            add_gaussian_noise(self.enu[:,2],0)))
        
        self.yaw_vf_wz_noise = np.column_stack((add_gaussian_noise(self.yaw_vf_wz[:,0],0),
                                                add_gaussian_noise(self.yaw_vf_wz[:,1],self.sigma_vf),
                                                add_gaussian_noise(self.yaw_vf_wz[:,2],self.sigma_wz)))

    
    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, and apply a Kalman filter to the noisy data (enu).
        """
	
	    # "TODO":	        

        # a) done -> dataloader
        # b) 
        lla_gps_gt_traj = build_LLA_GPS_trajectory(self.dataset)
        # timestamps done in c section

        # c) Transform GPS trajectory from global coordinates (LLA) to local coordinates (ENU→ x/y/z) 
        # in order to enable the Kalman filter to handle them.
        # done on the init function

        # Plot ground-truth GPS trajectory 
        graphs.plot_trajectory(lla_gps_gt_traj,title="Trajectory in World coordinates (LLA)",xlabel="LON",ylabel="LAT")
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_gt_global_coords')
        
        # Plot ground-truth GPS trajectory
        graphs.plot_trajectory(self.enu[:,:2],title="Trajectory in Local coordinates (ENU)",xlabel="East [m]",ylabel="North [m]")
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_gt_local_coords')
        # d) Add a gaussian noise to the ground-truth GPS data which will be used as noisy observations fed to the Kalman filter later
        # adding noise done -> init function of class
        
        # Plot the trajectory in local coordinates without noise (ENU) and observations noise (ENU+noise) on the same graph.
        graphs.plot_trajectory_comparison_with_and_without_noise(enu=self.enu[:,:2],enu_noise=self.enu_noise[:,:2])
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_gt&noise_local_coords')

        # e) Apply KalmanFilter
        sigma_n = 1
        k=3
        kf = KalmanFilter(enu_gt=self.enu,
                        enu_noise=self.enu_noise,
                        times = self.times,
                        sigma_xy=self.sigma_xy,
                        k=k,
                        sigma_n=sigma_n)

        kf_calib = KfCalib(kf=kf,
                        calib_folder=self.kf_calib_dir,
                        max_iter=30)

        best_params = kf_calib.run_calibration()
        
        # run the tuned filter
        kf.__setattr__('k',best_params['k'])
        kf.__setattr__('sigma_n',best_params['sigma_n'])
        locations_kf, covariance_kf = kf.run()
        

        # plot the estimate path
        graphs.plot_trajectory(locations_kf[:,[0,2]],title="Estimated Trajectory in Local coordiante (ENU)",xlabel="East [m]",ylabel="North [m]")
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_kf_traj')

        # plot gt, noise and estimate on the same graph
        graphs.plot_trajectory_comparison_with_and_without_noise(enu=self.enu[:,:2],enu_noise=self.enu_noise[:,:2],enu_predicted=locations_kf[:,[0,2]])
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_kf&gt&noise_traj')

        # calc_RMSE_maxE(locations_GT, locations_kf)
        rmse, maxE, error_x, error_y = kf.calc_RMSE_maxE(self.enu[:,:2],locations_kf[:,[0,2]])
        print(f"RMSE: {rmse}, maxE: {maxE}")        
        
        # Plot the estimated error of x-y values separately and corresponded sigma value along the trajectory
        x_cov, y_cov = covariance_kf[:,0,0], covariance_kf[:,2,2]
        graphs.plot_error((error_x,np.sqrt(x_cov)), (error_y,np.sqrt(y_cov)))
        graphs.show_graphs(folder=self.figs_dir, file_name='q1_kf_error')

        # build_animation (hint- graphs.build_animation)
        ani_traj = graphs.build_traj_animation(enu_gt=self.enu[:,:2], enu_noise=self.enu_noise[:,:2], enu_kf=locations_kf[:,[0,2]],title='KF Tajectory Animation')
        # animation that shows the covariances of the KF estimated path
        x_xy_xy_y_cov = (covariance_kf[:,[0,2],0:3:2]).reshape(covariance_kf[:,[0,2],0:3:2].shape[0],4)
        ani_cov = graphs.build_cov_animation(X_Y0=self.enu[:,:2], X_Y1=self.enu_noise[:,:2], X_Y2=locations_kf[:,[0,2]], x_xy_xy_y=x_xy_xy_y_cov,
                                            title='Covariance Confidence Ellipse animation',
                                            xlabel='X axis [meters]',
                                            ylabel='Y axis [meters]',
                                            label0='ENU GT',
                                            label1='ENU Noise',
                                            label2='ENU KF')

        # this animation that shows the covariances of the dead reckoning estimated path
        kf.__setattr__('is_dead_reckoning',True)
        locations_kf_drck, covariance_kf_drck = kf.run()
        
        # drck traj
        ani_traj_drck = graphs.build_traj_animation(enu_gt=self.enu[:,:2], enu_noise=self.enu_noise[:,:2], enu_kf=locations_kf_drck[:,[0,2]],title='KF Tajectory Animation')

        # drck cov
        x_xy_xy_y_cov_drck = (covariance_kf_drck[:,[0,2],0:3:2]).reshape(covariance_kf_drck[:,[0,2],0:3:2].shape[0],4)
        ani_cov_drck = graphs.build_cov_animation(X_Y0=self.enu[:,:2], X_Y1=self.enu_noise[:,:2], X_Y2=locations_kf_drck[:,[0,2]], x_xy_xy_y=x_xy_xy_y_cov_drck,
                                            title='Covariance Confidence Ellipse animation',
                                            xlabel='X axis [meters]',
                                            ylabel='Y axis [meters]',
                                            label0='ENU GT',
                                            label1='ENU Noise',
                                            label2='ENU KF')

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")
        graphs.save_animation(ani=ani_traj,basedir= self.ani_dir,file_name='q1_KF_traj_ani')
        graphs.save_animation(ani=ani_cov,basedir= self.ani_dir,file_name='q1_KF_cov_ani')
        graphs.save_animation(ani=ani_cov_drck,basedir= self.ani_dir,file_name='q1_KF_cov_drck_ani')

    def Q2(self):

        """
        That function runs the code of question 2 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, yaw rate, and velocities, and apply a Kalman filter to the noisy data.
        """
        # a) done -> dataloader
        # b) done

        # plot yaw, yaw rate and forward velocity
        graphs.plot_yaw_yaw_rate_fv(self.yaw_vf_wz[:,0],self.yaw_vf_wz[:,2],self.yaw_vf_wz[:,1])
        graphs.show_graphs(folder=self.figs_dir, file_name='q2_yaw_yawr_vf_gt_plot')

        # plot vf and wz with and without noise
        graphs.plot_vf_wz_with_and_without_noise(self.yaw_vf_wz,self.yaw_vf_wz_noise)
        graphs.show_graphs(folder=self.figs_dir, file_name='q2_yawr_vf_with_and_without_noise_plot')

        # G) Apply ExtendedKalmanFilter
        k=3
        sigma_rxy = 0.1
        sigma_rtheta = 0.01
        sigma_r = [sigma_rxy,sigma_rtheta]
        sigma_theta = 0.5
        ekf = ExtendedKalmanFilter(enu_gt=self.enu,
                                    enu_noise=self.enu_noise,
                                    yaw_vf_wz=self.yaw_vf_wz_noise,
                                    times=self.times,
                                    sigma_xy=self.sigma_xy,
                                    sigma_theta= sigma_theta,
                                    sigma_wz=self.sigma_wz,
                                    sigma_vf=self.sigma_vf,
                                    sigma_r=sigma_r,
                                    k=k)

        ekf_calib = EkfCalib(ekf=ekf,
                        calib_folder=self.ekf_calib_dir,
                        max_iter=100)

        best_params = ekf_calib.run_calibration()
        
        # run the tuned filter
        ekf.__setattr__('k',best_params['k'])
        ekf.__setattr__('sigma_theta',best_params['sigma_theta'])
        ekf.__setattr__('sigma_r',[best_params['sigma_rxy'],best_params['sigma_rtheta']])

        locations_ekf, covariance_ekf = ekf.run()
        x_y_theta_gt = np.concatenate([self.enu[:,:2],self.yaw_vf_wz[:,0][:,np.newaxis]],axis=1)
        rmse, maxE, error_x, error_y, error_yaw = ekf.calc_RMSE_maxE(x_y_theta_gt, locations_ekf)

        print(f"RMSE: {rmse}, maxE: {maxE}")        
        
        # plot gt, noise and estimate on the same graph
        graphs.plot_trajectory_comparison_with_and_without_noise(enu=self.enu[:,:2],enu_noise=self.enu_noise[:,:2],enu_predicted=locations_ekf[:,[0,1]])
        graphs.show_graphs(folder=self.figs_dir, file_name='q2_kf&gt&noise_traj')
        
        # Plot the estimated error of x-y-θ values separately and corresponded sigma value along the trajectory
        x_cov, y_cov, yaw_cov = covariance_ekf[:,0,0], covariance_ekf[:,1,1], covariance_ekf[:,2,2]
        graphs.plot_error((error_x,np.sqrt(x_cov)), (error_y,np.sqrt(y_cov)),(error_yaw,np.sqrt(yaw_cov)))
        graphs.show_graphs(folder=self.figs_dir, file_name='q2_ekf_error')

        # build_animation (hint- graphs.build_animation)
        ani_traj = graphs.build_traj_animation(enu_gt=self.enu[:,:2], enu_noise=self.enu_noise[:,:2], enu_kf=locations_ekf[:,:2],title='EKF Tajectory Animation')
        # animation that shows the covariances of the KF estimated path
        x_xy_xy_y_cov = (covariance_ekf[:,:2,:2]).reshape(covariance_ekf[:,:2,0:2].shape[0],4)
        ani_cov = graphs.build_cov_animation(X_Y0=self.enu[:,:2], X_Y1=self.enu_noise[:,:2], X_Y2=locations_ekf[:,:2], x_xy_xy_y=x_xy_xy_y_cov,
                                            title='Covariance Confidence Ellipse animation',
                                            xlabel='X axis [meters]',
                                            ylabel='Y axis [meters]',
                                            label0='ENU GT',
                                            label1='ENU Noise',
                                            label2='ENU EKF')

        # this animation that shows the covariances of the dead reckoning estimated path
        ekf.__setattr__('is_dead_reckoning',True)
        locations_ekf_drck, covariance_ekf_drck = ekf.run()

        # drck traj
        ani_traj_drck = graphs.build_traj_animation(enu_gt=self.enu[:,:2], enu_noise=self.enu_noise[:,:2], enu_kf=locations_ekf_drck[:,:2],title='EKF Tajectory Animation')

        # drck cov
        x_xy_xy_y_cov_drck = (covariance_ekf_drck[:,:2,:2]).reshape(covariance_ekf_drck[:,:2,:2].shape[0],4)
        ani_cov_drck = graphs.build_cov_animation(X_Y0=self.enu[:,:2], X_Y1=self.enu_noise[:,:2], X_Y2=locations_ekf_drck[:,:2], x_xy_xy_y=x_xy_xy_y_cov_drck,
                                            title='Covariance Confidence Ellipse animation',
                                            xlabel='X axis [meters]',
                                            ylabel='Y axis [meters]',
                                            label0='ENU GT',
                                            label1='ENU Noise',
                                            label2='ENU EKF')

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")
        graphs.save_animation(ani=ani_traj,basedir= self.ani_dir,file_name='q2_EKF_traj_ani')
        graphs.save_animation(ani=ani_cov,basedir= self.ani_dir,file_name='q2_EKF_cov_ani')
        graphs.save_animation(ani=ani_cov_drck,basedir= self.ani_dir,file_name='q2_EKF_cov_drck_ani')
        # print the maxE and RMSE

        # draw the trajectories

        # draw the error

        #v.	Plot the estimated error of x-y-θ values separately and corresponded sigma value along the trajectory
        
       
 
        # build_animation

            # animation that shows the covariances of the EKF estimated path

            # this animation that shows the covariances of the dead reckoning estimated path

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")

        #graphs.show_graphs()

    def get_odometry(self, sensor_data):
        """
        Args:
            sensor_data: map from a tuple (frame number, type) where type is either ‘odometry’ or ‘sensor’.
            Odometry data is given as a map containing values for ‘r1’, ‘t’ and ‘r2’ – the first angle, the translation and the second angle in the odometry model respectively.
            Sensor data is given as a map containing:
              - ‘id’ – a list of landmark ids (starting at 1, like in the landmarks structure)
              - ‘range’ – list of ranges, in order corresponding to the ids
              - ‘bearing’ – list of bearing angles in radians, in order corresponding to the ids

        Returns:
            numpy array of dim [num of frames X 3]
            first two components in each row are the x and y in meters
            the third component is the heading in radians
        """
        num_frames = len(sensor_data) // 2
        state = np.array([[0, 0, 0]], dtype=float).reshape(1, 3)
        for i in range(num_frames):
            curr_odometry = sensor_data[i, 'odometry']
            t = np.array([
                curr_odometry['t'] * np.cos(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['t'] * np.sin(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['r1'] + curr_odometry['r2']
            ]).reshape(3, 1)
            new_pos = state[-1, :].reshape(3, 1) + t
            state = np.concatenate([state, new_pos.reshape(1, 3)], axis=0)
        return state
    
        
    def Q3(self):

        """
        Runs the code for question 3 of the project
        Loads the odometry (robot motion) and sensor (landmarks) data supplied with the exercise
        Adds noise to the odometry data r1, trans and r2
        Uses the extended Kalman filter SLAM algorithm with the noisy odometry data to predict the path of the robot and
        the landmarks positions
        """

        #Pre-processing
        landmarks = self.dataset.load_landmarks()
        sensor_data_gt = self.dataset.load_sensor_data()
        state = self.get_odometry(sensor_data_gt)      
        #TODO
        sigma_x_y_theta = [1.5,1.5, 0.5]
        variance_r1_t_r2 = [0.01**2,0.1**2,0.01**2]
        variance_r_phi = [0.3**2,0.035**2]

        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        state_noised = self.get_odometry(sensor_data_noised)

        #TODO
        # plot trajectory 
        graphs.plot_trajectory(state[:,:2],"GT trajectory odom model","x-axis [m]","y-axis [m]")
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_gt_trajectory')

        # plot trajectory + noise
        graphs.plot_trajectory_with_noise(state[:,:2],state_noised[:,:2],
                                          "GT and noised trajectories odom model",
                                          "x-axis [m]",
                                          "y-axis [m]",
                                          "gt path",
                                          "noised path")
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_gt_with_noise')

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # KalmanFilter

        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)

        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

        rmse, maxE, error_x, error_y, error_yaw = ekf_slam.calc_RMSE_maxE(mu_arr_gt, mu_arr[:,:3])
        
        print(f"rmse: {rmse} | maxE: {maxE}")
        
        # draw the error for x, y and theta

        # Plot the estimated error ofof x-y-θ -#landmark values separately and corresponded sigma value along the trajectory

        # draw the error

        graphs.plot_trajectory_comparison(enu=mu_arr_gt[:,:2],enu_predicted=mu_arr[:,:2])
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_gt_vs_predicted')

        graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-$x_n$", "frame", "error", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_x')
        
        graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-$y_n$", "frame", "error", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_y')
        
        graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "$\\theta-\\theta_n$", 
                                 "frame", "error", "$\\theta-\\theta_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_theta')

        graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]), 
                                 "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_landmark1_x')

        graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]), 
                                 "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_landmark1_y')

        graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
                                 "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_landmark2_x')
        
        graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
                                 "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$", 
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))
        graphs.show_graphs(folder=self.figs_dir, file_name='q3_error_landmark2_y')

        ax.set_xlim([-2, 12])
        ax.set_ylim([-2, 12])
        
        ani = animation.ArtistAnimation(fig, frames, repeat=False)
        graphs.show_graphs()
        graphs.save_animation(ani=ani,basedir= self.ani_dir,file_name='q3_EKF_SLAM_traj_ani')

    def run(self):
        self.Q1()
        #self.Q2()
        #self.Q3()

