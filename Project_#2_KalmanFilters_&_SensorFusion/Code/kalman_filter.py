from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_state import plot_state
from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:
    """
    class for the implementation of Kalman filter
    """

    def __init__(self, enu_noise, times, sigma_xy, sigma_n, k=3,enu_gt=None, is_dead_reckoning=False):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
        """
        self.enu_gt = enu_gt # for calibration purposes
        self.enu_noise = enu_noise
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_n = sigma_n
        self.is_dead_reckoning = is_dead_reckoning
        self.k = k

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """


        # N = len(X_Y_GT[100:,0])
        error_x = X_Y_GT[:,0] - X_Y_est[:,0]
        error_y = X_Y_GT[:,1] - X_Y_est[:,1]

        # Calculate the squared error between the ground truth and estimated values
        squared_error_x = error_x ** 2
        squared_error_y = error_y ** 2

        min_idx = 100
        # Calculate the mean of the squared error to get the RMSE
        rmse = np.sqrt(np.mean(squared_error_x[min_idx:] + squared_error_y[min_idx:]))

        # Calculate the max error between the ground truth and estimated values
        maxE = np.max(np.abs(error_x[min_idx:]) + np.abs(error_y[min_idx:]))

        return rmse, maxE, error_x, error_y
    
    #TODO
    def run(self):
        """
        Runs the Kalman filter

        outputs: enu_kf, covs
        """
        
        # init time

        time = 0
        # get initial belief
        mu0, cov0 = self.initialize()

        mu_filtered = [mu0]
        cov_filtered = [cov0]

        C = np.array([[1,0,0,0],
                        [0,0,1,0]])

        # iterate over the delta time
        for i in range(1,len(self.times)):
            
            if i==1:
                mu_prev = mu0
                cov_prev = cov0

            # get time difference and add to time
            del_t = self.times[i] - self.times[i-1]
            time += del_t

            # set the state transition matrix
            A_t = np.array([[1,del_t,0,0],
                            [0,1,0,0],
                            [0,0,1,del_t],
                            [0,0,0,1]])
            
            # set the process noise 
            R_t = (self.sigma_n**2)*np.array([[del_t,0,0,0],
                                            [0,1,0,0],
                                            [0,0,del_t,0],
                                            [0,0,0,1]])
            
            
            # set the observations noise 
            Q_t = np.array([[self.sigma_xy**2,  0], 
                            [   0,     self.sigma_xy**2]]) 

            # filtering loop
            # prediction step
            mu_t_pred , cov_t_pred = self.prediction(mu_prev,cov_prev,A_t,R_t)
            # calc kalman gain
            if self.is_dead_reckoning and time >= 5:
                k_gain = np.zeros((4,2))
            else:
                k_gain = self.get_kalman_gain(cov_t_pred,C,Q_t)
            # correction step
            mu_t, cov_t = self.correction(mu_t_pred,cov_t_pred,k_gain,C,step=i)

            # append
            mu_filtered.append(mu_t)
            cov_filtered.append(cov_t)

            # update prev state and cov
            mu_prev, cov_prev = mu_t, cov_t
        
        return np.array(mu_filtered), np.array(cov_filtered)




    def initialize(self):
        vx0,vy0 = 0,0
        x0,y0 = self.enu_noise[0,0],self.enu_noise[0,1]
        mu0 = np.array([x0,vx0,y0,vy0]).T


        cov0 = np.array([[self.k*self.sigma_xy**2,0,0,0], # check sigma_v
                            [0,100,0,0],
                            [0,0,self.k*self.sigma_xy**2,0],
                            [0,0,0,100]]) 

        return mu0, cov0

    def prediction(self,mu_prev,cov_prev,A_t,R_t):
        mu_pred = self.get_state(mu_prev,A_t)
        cov_pred = self.get_cov(cov_prev,A_t,R_t)

        return mu_pred, cov_pred

    def get_state(self, mu_prev, A_t):
        return (A_t.dot(mu_prev)).T

    def get_cov(self,cov_prev,A_t,R_t):
        return (A_t.dot(cov_prev)).dot(A_t.T) + R_t

    def get_kalman_gain(self,cov_t_pred,C,Q_t):
        # Calculate the Kalman gain K
        K_gain = cov_t_pred.dot(C.T).dot(np.linalg.inv(C.dot(cov_t_pred).dot(C.T) + Q_t))
        return K_gain


    def correction(self,mu_t_pred,cov_t_pred,k_gain,c,step):
        z_t = self.get_meas(step)
        # Update the estimate and error covariance matrices
        mu_t_cor =  mu_t_pred + k_gain.dot(z_t - c.dot(mu_t_pred))
        cov_t_cor = (np.eye(cov_t_pred.shape[0]) - k_gain.dot(c)).dot(cov_t_pred)

        return mu_t_cor, cov_t_cor

    def get_meas(self,step): #mu_t_pred,c):
        # return (c.dot(mu_t_pred)).T
        return np.array([self.enu_noise[step,0],self.enu_noise[step,1]]).T #check if velocity need to be 0 ? 




class ExtendedKalmanFilter:
    """
    class for the implementation of the extended Kalman filter
    """
    def __init__(self, enu_noise, yaw_vf_wz, times, sigma_xy, sigma_theta, sigma_vf, sigma_wz,sigma_r, k,enu_gt=None, is_dead_reckoning=False, dead_reckoning_start_sec=5.0):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            yaw_vf_wz: the yaw, forward velocity and angular change rate to be used (either non noisy or noisy, depending on the question)
            sigma_theta: sigma of the heading
            sigma_vf: sigma of the forward velocity
            sigma_wz: sigma of the angular change rate
            k: hyper parameter to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
            dead_reckoning_start_sec: from what second do we start applying dead reckoning, used for experimentation only
        """
        self.enu_gt = enu_gt # for calibration purposes
        self.enu_noise = enu_noise
        self.yaw_vf_wz = yaw_vf_wz
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_theta = sigma_theta
        self.sigma_vf = sigma_vf
        self.sigma_wz = sigma_wz
        self.sigma_r = sigma_r
        self.k = k
        self.is_dead_reckoning = is_dead_reckoning
        self.dead_reckoning_start_sec = dead_reckoning_start_sec


    #TODO
    @staticmethod
    def calc_RMSE_maxE(X_Y_theta_GT, X_Y_theta_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """

        # N = len(X_Y_theta_GT[100:,0])
        error_x = X_Y_theta_GT[:,0] - X_Y_theta_est[:,0]
        error_y = X_Y_theta_GT[:,1] - X_Y_theta_est[:,1]
        error_yaw = X_Y_theta_GT[:,2] - X_Y_theta_est[:,2]

        # Calculate the squared error between the ground truth and estimated values
        squared_error_x = error_x ** 2
        squared_error_y = error_y ** 2

        min_idx = 100
        # Calculate the mean of the squared error to get the RMSE
        rmse = np.sqrt(np.mean(squared_error_x[min_idx:] + squared_error_y[min_idx:]))

        # Calculate the max error between the ground truth and estimated values
        maxE = np.max(np.abs(error_x[min_idx:]) + np.abs(error_y[min_idx:]))

        return rmse, maxE, error_x, error_y, error_yaw
    

    def run(self):
        """
        Runs the extended Kalman filter
        outputs: enu_ekf, covs
        """
        # init time

        time = 0
        # get initial belief
        mu0, cov0 = self.initialize()
        
        mu_filtered = [mu0]
        cov_filtered = [cov0]

        H = np.array([[1,0,0],
                        [0,1,0]])

        # iterate over the delta time
        for i in range(1,len(self.times)):
            
            if i==1:
                mu_prev = mu0
                cov_prev = cov0

            # get time difference and add to time
            dt = self.times[i] - self.times[i-1]
            time += dt

            # set the state transition matrix
            ut = np.array([self.yaw_vf_wz[i,1],self.yaw_vf_wz[i,2]])
            vt = ut[0]
            wt = ut[1]
            
            G_t = np.array([[1,0,-(vt/wt)*np.cos(mu_prev[2])+(vt/wt)*np.cos(mu_prev[2]+wt*dt)],
                            [0,1,-(vt/wt)*np.sin(mu_prev[2])+(vt/wt)*np.sin(mu_prev[2]+wt*dt)],
                            [0,0,1]])
            

            V_t = np.array([[-(1/wt)*np.sin(mu_prev[2]) + (1/wt)*np.sin(mu_prev[2]+wt*dt), (vt/(wt**2))*np.sin(mu_prev[2]) - (vt/(wt**2))*np.sin(mu_prev[2]+wt*dt) + (vt/wt)*np.cos(mu_prev[2]+wt*dt)*dt],
                            [(1/wt)*np.cos(mu_prev[2]) - (1/wt)*np.cos(mu_prev[2]+wt*dt), -(vt/(wt**2))*np.cos(mu_prev[2]) + (vt/(wt**2))*np.cos(mu_prev[2]+wt*dt) + (vt/wt)*np.sin(mu_prev[2]+wt*dt)*dt],
                            [0,dt]]) 


            R_t1 = np.array([[self.sigma_vf**2,0],
                                            [0,self.sigma_wz**2]]) 
            
            R_t2 = np.diag([self.sigma_r[0],self.sigma_r[0],self.sigma_r[1]])

            # set the observations noise 
            Q_t = np.array([[self.sigma_xy**2,  0], 
                            [   0,     self.sigma_xy**2]]) 

            # filtering loop
            # prediction step
            mu_t_pred , cov_t_pred = self.prediction(self.g(ut,mu_prev,dt),cov_prev,G_t,V_t,R_t1,R_t2)
            mu_t_pred[2] = normalize_angle(mu_t_pred[2])
            # calc kalman gain
            if self.is_dead_reckoning and time >= 5:
                k_gain = np.zeros((3,2))
            else:
                k_gain = self.get_kalman_gain(cov_t_pred,H,Q_t)
            # correction step
            mu_t, cov_t = self.correction(mu_t_pred,cov_t_pred,k_gain,H,step=i)
            mu_t[2] = normalize_angle(mu_t[2])
            # append
            mu_filtered.append(mu_t)
            cov_filtered.append(cov_t)

            # update prev state and cov
            mu_prev, cov_prev = mu_t, cov_t

        mu = np.array(mu_filtered)
        cov = np.array(cov_filtered)
        return mu , cov 



    def initialize(self):
        theta0 = self.yaw_vf_wz[0,0]
        x0,y0 = self.enu_noise[0,0],self.enu_noise[0,1]
        mu0 = np.array([x0,y0,theta0]).T
        mu0[2] = normalize_angle(mu0[2])

        cov0 = np.array([[self.k*self.sigma_xy**2,0,0], # check sigma_v
                            [0,self.k*self.sigma_xy**2,0],
                            [0,0,self.k*self.sigma_theta**2]]) 

        return mu0, cov0

    def prediction(self,g,cov_prev,G_t,V_t,R_t1,R_t2):
        mu_pred = self.get_state(g)
        cov_pred = self.get_cov(cov_prev,G_t,V_t,R_t1,R_t2)

        return mu_pred, cov_pred
    
    def get_state(self,g):
        return g
    
    def g(self,ut, mu_prev,dt):
        vt = ut[0]
        wt = ut[1]
        mu = mu_prev + np.array([-(vt/wt)*np.sin(mu_prev[2])+(vt/wt)*np.sin(mu_prev[2]+wt*dt),
                                (vt/wt)*np.cos(mu_prev[2])-(vt/wt)*np.cos(mu_prev[2]+wt*dt),
                                wt*dt]).T
        return mu


    def get_cov(self,cov_prev,G_t,V_t,R_t1,R_t2):
        return (G_t.dot(cov_prev)).dot(G_t.T) + (V_t.dot(R_t1)).dot(V_t.T) + R_t2

    def get_kalman_gain(self,cov_t_pred,H,Q_t):
        # Calculate the Kalman gain K
        K_gain = cov_t_pred.dot(H.T).dot(np.linalg.inv(H.dot(cov_t_pred).dot(H.T) + Q_t))
        return K_gain

    def correction(self,mu_t_pred,cov_t_pred,k_gain,H,step):
        z_t = self.get_meas(step)
        # Update the estimate and error covariance matrices
        mu_t_cor =  mu_t_pred + k_gain.dot(z_t - H.dot(mu_t_pred))
        cov_t_cor = (np.eye(cov_t_pred.shape[0]) - k_gain.dot(H)).dot(cov_t_pred)

        return mu_t_cor, cov_t_cor

    def get_meas(self,step): #mu_t_pred,c):
        # return (c.dot(mu_t_pred)).T
        return np.array([self.enu_noise[step,0],self.enu_noise[step,1]]).T #check if velocity need to be 0 ? 



class ExtendedKalmanFilterSLAM:
    
    
    def __init__(self, sigma_x_y_theta, variance_r1_t_r2, variance_r_phi):
        """
        Args:
            variance_x_y_theta: variance in x, y and theta respectively
            variance_r1_t_r2: variance in rotation1, translation and rotation2 respectively
            variance_r_phi: variance in the range and bearing
        """
        #TODO
        self.sigma_x_y_theta = sigma_x_y_theta
        self.variance_r_phi = variance_r_phi
        self.R_x = np.diag(variance_r1_t_r2)


    def predict(self, mu_prev, sigma_prev, u, N):
        # Perform the prediction step of the EKF
        #u[0]=translation, u[1]=rotation1, u[2]=rotation2 <<--- not true

        delta_trans, delta_rot1, delta_rot2 = u['t'],u['r1'],u['r2']
        theta_prev = mu_prev[2]
        
        #TODO
        # mu prediction
        F = np.zeros((3,3+2*N))
        F[:,:3] = np.eye(3)

        motion_model = np.array([delta_trans * np.cos(theta_prev + delta_rot1),
                                delta_trans * np.sin(theta_prev + delta_rot1),
                                delta_rot1 + delta_rot2]).T
        
        mu_est = mu_prev + F.T.dot(motion_model)


        G_x = np.eye(3) + (np.array([[0,0,-delta_trans*np.sin(theta_prev+delta_rot1)],
                                    [0,0,delta_trans*np.cos(theta_prev+delta_rot1)],
                                    [0,0,0]])) #jacobian of the motion
        
        G = np.zeros((3+2*N,3+2*N))
        G[:3,:3] = G_x.copy()
        G[3:,3:] = np.eye(2*N)

        V = np.array([[-delta_trans*np.sin(theta_prev+delta_rot1),np.cos(theta_prev+delta_rot1),0],
                      [delta_trans*np.cos(theta_prev+delta_rot1),np.sin(theta_prev+delta_rot1),0],
                      [1,0,1]])
        
        R_tx = V.dot(self.R_x.dot(V.T))
        R_t = F.T.dot(R_tx.dot(F))
        sigma_est = G.dot(sigma_prev.dot(G.T)) + R_t
        return mu_est, sigma_est
    
    def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
        # Perform filter update (correction) for each odometry-observation pair read from the data file.
        mu = mu_pred.copy()
        sigma = sigma_pred.copy()
        theta = mu[2]

        m = len(z["id"])
        Z = np.zeros(2 * m)
        z_hat = np.zeros(2 * m)
        H = None
        
        for idx in range(m):
            j = z["id"][idx] - 1
            r = z["range"][idx]
            phi = z["bearing"][idx]


            mu_j_x_idx = 3 + j*2
            mu_j_y_idx = 4 + j*2
            Z_j_x_idx = idx*2
            Z_j_y_idx = 1 + idx*2
            
            if observed_landmarks[j] == False:
                mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
                observed_landmarks[j] = True
                
            Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])
            
            delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
            q = delta.dot(delta)

            #TODO
            z_hat[Z_j_x_idx : Z_j_y_idx + 1] = [np.sqrt(q),np.arctan2(delta[1],delta[0])-theta]
            
            I = np.diag(5*[1])
            F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))
            
            #TODO
            Hi = (1/q)*np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                                [delta[1], -delta[0], -q, -delta[1], delta[0]]])

            Hi = Hi.dot(F_j)
            
            if H is None:
                H = Hi.copy()
            else:
                H = np.vstack((H, Hi))

        #TODO
        Q = np.diag(self.variance_r_phi*(int(H.shape[0]/2)))
        ##S = ??
        K = sigma.dot(H.T.dot(np.linalg.inv((H.dot(sigma.dot(H.T))) + Q)))

        #TODO
        diff = Z - z_hat 
        diff[1::2] = normalize_angles_array(diff[1::2])
        
        mu = mu + K.dot(diff)
        mu[2] = normalize_angle(mu[2])

        #TODO
        sigma = (np.eye(21) - (K.dot(H))).dot(sigma) 

        # Remember to normalize the bearings after subtracting!
        # (hint: use the normalize_all_bearings function available in tools)

        # Finish the correction step by computing the new mu and sigma.
        # Normalize theta in the robot pose.

        
        return mu, sigma, observed_landmarks


    def init_mu_sigma(self,marks_len,init_inf_val):
        mu0 = np.zeros(3+2*marks_len)

        sigma_x_y_theta0 = np.diag(np.power(self.sigma_x_y_theta,2))
        sigma_marks0 = np.diag([init_inf_val]*(2*marks_len))
        sigma_prev = np.zeros((3+2*marks_len,3+2*marks_len))
        sigma_prev[:3,:3] = sigma_x_y_theta0.copy()
        sigma_prev[3:,3:] = sigma_marks0.copy()

        return mu0.T, sigma_prev



    def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
        # Get the number of landmarks in the map
        N = len(landmarks)
        
        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution

        #TODO
        init_inf_val = 5
        mu0, sigma_prev = self.init_mu_sigma(N,init_inf_val)
        mu_arr = [mu0]

        #sigma for analysis graph sigma_x_y_t + select 2 landmarks
        landmark1_ind=3
        landmark2_ind=7

        Index=[0,1,2,landmark1_ind,landmark1_ind+1,landmark2_ind,landmark2_ind+1]
        sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index,Index].copy()
        
        observed_landmarks = np.zeros(N, dtype=bool)
        
        sensor_data_count = int(len(sensor_data_noised) / 2)
        frames = []
        
        mu_arr_gt = np.array([[0, 0, 0]])
        
        for idx in range(sensor_data_count):
            mu_prev = mu_arr[-1]
            
            u = sensor_data_noised[(idx, "odometry")]
            # predict
            mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)
            # update (correct)
            mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)
            
            mu_arr = np.vstack((mu_arr, mu))
            sigma_prev = sigma.copy()
            sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index,Index].copy()))
            
            delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
            delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
            delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]

            calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
            calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)

            theta = delta_r1_gt + delta_r2_gt

            theta_prev = mu_arr_gt[-1,2]
            mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))
            
            frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])
            
            frames.append(frame)
        
        a=1
        return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2

    @staticmethod
    def calc_RMSE_maxE(X_Y_theta_GT, X_Y_theta_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """


        error_x = X_Y_theta_GT[:,0] - X_Y_theta_est[:,0]
        error_y = X_Y_theta_GT[:,1] - X_Y_theta_est[:,1]
        error_yaw = X_Y_theta_GT[:,2] - X_Y_theta_est[:,2]

        # Calculate the squared error between the ground truth and estimated values
        squared_error_x = error_x ** 2
        squared_error_y = error_y ** 2

        min_idx = 20
        # Calculate the mean of the squared error to get the RMSE
        rmse = np.sqrt(np.mean(squared_error_x[min_idx:] + squared_error_y[min_idx:]))

        # Calculate the max error between the ground truth and estimated values
        maxE = np.max(np.abs(error_x[min_idx:]) + np.abs(error_y[min_idx:]))

        return rmse, maxE, error_x, error_y, error_yaw
