from utils.ellipse import draw_prob_ellipse
from utils.draw_robot import draw_robot
import numpy as np
import matplotlib.pyplot as plt

def plot_state(ax, gt_arr, mu_arr, sigma, landmarks, observedLandmarks, z): #, window)
    # Visualizes the state of the EKF SLAM algorithm.
    #
    # The resulting plot displays the following information:
    # - map ground truth (black +'s)
    # - current robot pose estimate (red)
    # - current landmark pose estimates (blue)
    # - visualization of the observations made at this time step (line between robot and landmark)
    frame = []
    
    mu = mu_arr[-1,:].copy()
    second_val = lambda x: x[1]
    L = np.array(list(map(second_val, list(landmarks.items()))))
    # L = np.array(landmarks)
    frame.append(draw_prob_ellipse(ax, mu[0:3], sigma[0:3, 0:3], 0.6, 'r'))
    
    ax_plot, = ax.plot(L[:,0], L[:,1], 'k+', markersize=5, linewidth=2)
    frame.append(ax_plot)
    
    for i in range(len(observedLandmarks)):
        if observedLandmarks[i]:
            ax_plot, = ax.plot(mu[2*i+3], mu[2*i+4], 'bo', markersize=5, linewidth=2)
            frame.append(ax_plot)
            frame.append(draw_prob_ellipse(ax, mu[2*i+3: 2*i+5], sigma[2*i+3: 2*i+5, 2*i+3: 2*i+5], 0.6, 'b'))
    
    for i in range(len(z["id"])):
        mX = mu[2*(z["id"][i]-1) + 3]
        mY = mu[2*(z["id"][i]-1) + 4]
        
        ax_plot, = ax.plot([mu[0], mX], [mu[1], mY], color='k', linewidth=1)
        frame.append(ax_plot)

    h1, h2 = draw_robot(ax, mu, 'r', 3, 0.3, 0.3)
    frame.append(h1)
    frame.append(h2)
    
    ax_plot, = ax.plot(mu_arr[:,0], mu_arr[:,1], c='g', linewidth=2)
    frame.append(ax_plot)
    
    ax_plot, = ax.plot(gt_arr[:,0], gt_arr[:,1], c='orange', linewidth=2)
    frame.append(ax_plot)
    
    return frame
