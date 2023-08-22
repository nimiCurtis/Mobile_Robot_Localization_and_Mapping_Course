import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import numpy as np
from utils.misc_tools import error_ellipse
import io
from PIL import Image

def plot_yaw_yaw_rate_fv(yaw, yaw_rate, fv):
    """
    plots the yaw, yaw rate, and forward velocity
    Args:
        yaw: heading
        yaw_rate: change in heading
        fv: forward velocity
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(range(yaw.shape[0]), yaw, 'b')
    ax1.plot(range(yaw_rate.shape[0]), yaw_rate, 'r')
    ax1.set_title("Yaw and yaw change rate per frame", fontsize=16)
    ax1.set_xlabel("Frame #", fontsize=16)
    ax1.set_ylabel("Yaw [rad]", fontsize=16)
    ax1.legend(["Yaw [rad]", "Yaw change rate [rad/s]"], prop={"size": 14}, loc="best")

    ax2.plot(range(yaw_rate.shape[0]), yaw_rate, 'b')
    ax2.set_title("Yaw change rate per frame (isolated)", fontsize=16)
    ax2.set_xlabel("Frame #", fontsize=16)
    ax2.set_ylabel("Yaw [rad/s]", fontsize=16)

    ax3.plot(range(fv.shape[0]), fv, 'b')
    ax3.set_title("Forward velocity per frame", fontsize=16)
    ax3.set_xlabel("Frame #", fontsize=16)
    ax3.set_ylabel("Forward Velocity [m/s]", fontsize=16)


def plot_vf_wz_with_and_without_noise(yaw_vf_wz, yaw_vf_wz_noise):
    """
    plots forward velocity and angular change rate with and without noise
    Args:
        yaw_vf_wz: numpy array of triplets (yaw, forward velocity, angular change rate)
        yaw_vf_wz_noise: similar to yaw_vf_wz but contains noise
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(range(yaw_vf_wz_noise.shape[0]), yaw_vf_wz_noise[:, 1], 'r')
    ax1.plot(range(yaw_vf_wz.shape[0]), yaw_vf_wz[:, 1], 'b')
    ax1.set_title("Forward velocity per frame and the added noise", fontsize=16)
    ax1.set_xlabel("Frame #", fontsize=16)
    ax1.set_ylabel("Forward Velocity [m/s]", fontsize=16)
    ax1.legend(["Noised forward velocity", "Forward velocity ground truth"], prop={"size": 14}, loc="best")

    ax2.plot(range(yaw_vf_wz_noise.shape[0]), yaw_vf_wz_noise[:, 2], 'r')
    ax2.plot(range(yaw_vf_wz.shape[0]), yaw_vf_wz[:, 2], 'b')
    ax2.set_title("Yaw change rate per frame and the added noise", fontsize=16)
    ax2.set_xlabel("Frame #", fontsize=16)
    ax2.set_ylabel("Yaw change rate [rad/s]", fontsize=16)
    ax2.legend(["Noised yaw change rate", "Yaw change rate ground truth"], prop={"size": 14}, loc="best")

def plot_error(err_cov_x, err_cov_y, err_cov_yaw=None):
    """
    Plots the error for x, y and their predicted sigma values for each frame
    optionally, the another graph is plotted with the same information for the heading
    Args:
        err_cov_x: tuple of (numpy array with difference in position across the x axis, predicted sigma in x axis)
        err_cov_y: tuple of (numpy array with difference in position across the y axis, predicted sigma in y axis)
        err_cov_yaw: optional, tuple of (numpy array with normalized difference in heading, predicted sigma in heading)
    """
    if err_cov_yaw:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    err_x, cov_x = err_cov_x
    ax1.plot(range(err_x.shape[0]), err_x, 'b')
    ax1.plot(range(len(cov_x)), cov_x, 'r')
    ax1.plot(range(len(cov_x)), [-a for a in cov_x], 'r')
    ax1.set_title(
        "Error throughout the path along the x axis",
        fontsize=16)
    ax1.set_xlabel("Frame #", fontsize=16)
    ax1.set_ylabel("Error (x_gt - x_predicted) [meters]", fontsize=16)
    ax1.legend(["Error (x_gt - x_predicted) [meters]", "Predicted variance"], prop={"size": 14}, loc="best")

    err_y, cov_y = err_cov_y
    ax2.plot(range(err_y.shape[0]), err_y, 'b')
    ax2.plot(range(len(cov_y)), cov_y, 'r')
    ax2.plot(range(len(cov_y)), [-a for a in cov_y], 'r')
    ax2.set_title(
        "Error throughout the path along the y axis",
        fontsize=16)
    ax2.set_xlabel("Frame #", fontsize=16)
    ax2.set_ylabel("Error (y_gt - y_predicted) [meters]", fontsize=16)
    ax2.legend(["Error (y_gt - y_predicted) [meters]", "Predicted variance"], prop={"size": 14}, loc="best")

    if err_cov_yaw:
        err_yaw, cov_yaw = err_cov_yaw
        ax3.plot(range(err_yaw.shape[0]), err_yaw, 'b')
        ax3.plot(range(len(cov_yaw)), cov_yaw, 'r')
        ax3.plot(range(len(cov_yaw)), [-a for a in cov_yaw], 'r')
        ax3.set_title(
            "Error throughout the path in yaw",
            fontsize=16)
        ax3.set_xlabel("Frame #", fontsize=16)
        ax3.set_ylabel("Error (yaw_gt - yaw_predicted) [rad]", fontsize=16)
        ax3.set_ylim([-3,3])
        ax3.legend(["Error (yaw_gt - yaw_predicted) [rad]", "Predicted variance"], prop={"size": 14}, loc="best")


def plot_trajectory_comparison_with_and_without_noise(enu, enu_noise, enu_predicted=None):
    """
    plots xy, en or ll in one graph and z, u or a (respectively) in a second graph
    Args:
        enu: xyz or enu or lla
        enu_noise: xyz or enu or lla with noise
        enu_predicted: optionally, xyz or enu or lla after correction
    """
    fig, ax = plt.subplots()
    ax.plot(enu[:, 0], enu[:, 1], 'b')
    ax.scatter(enu_noise[:, 0], enu_noise[:, 1] ,c='r',marker='.')
    is_predicted_enu_given = type(enu_predicted) is np.matrix or type(enu_predicted) is np.ndarray
    if is_predicted_enu_given:
        ax.plot(enu_predicted[:, 0], enu_predicted[:, 1], 'g')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Comparison of the ENU path with and w/o noise" + " and the predicted path" if is_predicted_enu_given else "", fontsize=16)
    ax.set_xlabel("East [meters]", fontsize=16)
    ax.set_ylabel("North [meters]", fontsize=16)
    if is_predicted_enu_given:
        ax.legend(["Ground truth trajectory", "Noised trajectory (mean 0, std 3 meters)", "Predicted trajectory"], prop={"size": 14}, loc="best")
    else:
        ax.legend(["Ground truth trajectory", "Noised trajectory (mean 0, std 3 meters)"], prop={"size": 14}, loc="best")


def plot_trajectory_comparison(enu, enu_predicted):
    """
    Plots a comparison between the ground trajectory and the predicted one
    Args:
        enu: ground truth enu data
        enu_predicted: predicted enu data
    """
    fig, ax = plt.subplots()
    ax.plot(enu[:, 0], enu[:, 1], 'b')
    ax.plot(enu_predicted[:, 0], enu_predicted[:, 1], 'g')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Comparison of the ground truth trajectory and the predicted trajectory", fontsize=16)
    ax.set_xlabel("East [meters]", fontsize=16)
    ax.set_ylabel("North [meters]", fontsize=16)
    ax.legend(["Ground truth trajectory", "Predicted trajectory"], prop={"size": 14}, loc="best")


def plot_trajectory_comparison_dead_reckoning(enu, enu_predicted, enu_dead_reckoning):
    """
    Plots the ground truth path, the predicted path and the predicted dead reckoning path
    Args:
        enu: numpy array where the rows are (x, y) it may contain the heading too without effect
        enu_predicted: numpy array where the rows are (x, y) it may contain the heading too withou
                       effect which represents the predicted path
        enu_dead_reckoning: numpy array where the rows are (x, y) it may contain the heading too without effect
                            which represents the predicted dead reckoning path
    """
    fig, ax = plt.subplots()
    ax.plot(enu[:, 0], enu[:, 1], 'b')
    ax.plot(enu_predicted[:, 0], enu_predicted[:, 1], 'g')
    ax.plot(enu_dead_reckoning[:, 0], enu_dead_reckoning[:, 1], 'r')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Comparison of the ground truth trajectory, the predicted trajectory and the dead reckoning trajectory", fontsize=16)
    ax.set_xlabel("East [meters]", fontsize=16)
    ax.set_ylabel("North [meters]", fontsize=16)
    ax.legend(["Ground truth trajectory", "Predicted trajectory", "Dead Reckoning"], prop={"size": 14}, loc="best")


def plot_trajectory(pos_xy, title, xlabel, ylabel):
    """
    Plots a single trajectory
    Args:
        pos_xy: xy data
        title, xlabel, ylabel: labels for the graph
    """
    fig, ax = plt.subplots()
    ax.plot(pos_xy[:, 0], pos_xy[:, 1], 'b')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)


def plot_trajectory_with_noise(pos_xy_gt, pos_xy_noise, title, xlabel, ylabel, legend_gt, legend_noise):
    """
    Plots a ground truth trajectory and a noisy trajectory for comparison
    Args:
        pos_xy: xy data
        pos_xy_noise: xy data
        title, xlabel, ylabel: labels for the graph
        legend_gt - legend text for the ground truth
        legend_noise - legend text for the noisy path
    """
    fig, ax = plt.subplots()
    ax.plot(pos_xy_gt[:, 0], pos_xy_gt[:, 1], 'b')
    ax.scatter(pos_xy_noise[:, 0], pos_xy_noise[:, 1],c='r',marker='.')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend([legend_gt, legend_noise], prop={"size": 14}, loc="best")


def plot_trajectory_and_height(locations, title1, xlabel1, ylabel1, title2, xlabel2, ylabel2):
    """
    Plots xy, en or ll in one graph and z, u or a in a second graph
    Args:
        locations: xyz or enu or lla
        title1, xlabel1, ylabel1: labels for the xy, en or ll graph
        title2, xlabel2, ylabel2: labels for the height (z, u or a) graph
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(locations[:, 0], locations[:, 1], 'b')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(title1, fontsize=16)
    ax1.set_xlabel(xlabel1, fontsize=16)
    ax1.set_ylabel(ylabel1, fontsize=16)

    ax2.plot(range(0, locations.shape[0]), locations[:, 2], color='red', linestyle='solid', markersize=1)
    ax2.set_title(title2, fontsize=16)
    ax2.set_xlabel(xlabel2, fontsize=16)
    ax2.set_ylabel(ylabel2, fontsize=16)


def show_graphs(folder=None, file_name=None):
    """
    Saves or shows the current plot
    Args:
        folder: optional, must be provided with file_name too, the image will be saved to this path with the given file name
        file_name: optional, must be provided with folder too, the image will be saved to this path with the given file name
    """
    if not folder or not file_name:
        plt.show()
    else:
        file_name = "{}/{}.png".format(folder, file_name)
        figure = plt.gcf()  # get current figure
        number_of_subplots_in_figure = len(plt.gcf().get_axes())
        figure.set_size_inches(number_of_subplots_in_figure * 18, 18)
        ram = io.BytesIO()
        plt.savefig(ram, format='png', dpi=100)
        ram.seek(0)
        im = Image.open(ram)
        im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
        im2.save(file_name, format='PNG')
        plt.close(figure)
    plt.close('all')
    
def save_animation(ani, basedir, file_name):
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='pearlofe'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer, dpi=100)
    print("Animation saved")

def plot_single_graph(X_Y, title, xlabel, ylabel, label, is_scatter=False, sigma=None):
    """
    That function plots a single graph

    Args:
        X_Y (np.ndarray): array of values X and Y, array shape [N, 2]
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label (str): sets legend's label value
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_scatter:
        plt.scatter(np.arange(X_Y.shape[0]), X_Y, s=1, label=label, c="b")
        if sigma is not None:
            plt.plot(np.arange(X_Y.shape[0]), sigma, c="orange")
            plt.plot(np.arange(X_Y.shape[0]), -sigma, c="orange")
    elif len(X_Y.shape) ==1:
        plt.plot(np.arange(X_Y.shape[0]), X_Y, label=label)
    else:
        plt.plot(X_Y[:, 0], X_Y[:, 1], label=label)
    
    plt.legend()


def plot_graph_and_scatter(X_Y0, X_Y1, title, xlabel, ylabel, label0, label1, color0='b', color1='r', point_size=1):
    """
    That function plots two graphs, plot and scatter

    Args:
        X_Y0 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 0
        X_Y1 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 1
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label0 (str): sets legend's label value of graph 0
        label1 (str): sets legend's label value of graph 1
        color0 (str): color of graph0
        color1 (str): color of graph1
        point_size(float): size of scatter points
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], label=label0, c=color0)
    plt.scatter(X_Y1[:, 0], X_Y1[:, 1], label=label1, s=point_size, c=color1)
    plt.legend()


def plot_four_graphs(X_values, Y0, Y1, Y2, Y3, title, xlabel, ylabel, label0, label1, label2, label3):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_values, Y0, label=label0)
    plt.plot(X_values, Y1, label=label1)
    plt.plot(X_values, Y2, label=label2)
    plt.plot(X_values, Y3, label=label3)
    plt.legend()


def plot_three_graphs(X_Y0, X_Y1, X_Y2, title, xlabel, ylabel, label0, label1, label2):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_Y0[:, 0], X_Y0[:, 1], 'b-', label=label0)
    plt.plot(X_Y1[:, 0], X_Y1[:, 1], 'g:', label=label1)
    plt.plot(X_Y2[:, 0], X_Y2[:, 1], 'r--', label=label2)
    plt.legend()


def build_cov_animation(X_Y0, X_Y1, X_Y2, x_xy_xy_y, title, xlabel, ylabel, label0, label1, label2):
    
    fig = plt.figure()
    fig.set_size_inches(18, 18)
    ax = fig.add_subplot(1, 1, 1)
    print("Creating animation")
    
    x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
    val0, = plt.plot([], [], 'b-', animated=True, label=label0)
    val1, = plt.plot([], [], 'g:', animated=True, label=label1)
    val2, = plt.plot([], [], 'r--', animated=True, label=label2)
    val3 = Ellipse(xy=(0,0), width=0, height=0, angle=0, animated=True)
    
    ax.add_patch(val3)
    plt.legend()
    
    values = np.hstack((X_Y0, X_Y1, X_Y2, x_xy_xy_y))
    
    def init():
        margin = 10
        x_min = np.min(X_Y0[:, 0]) - margin
        x_max = np.max(X_Y0[:, 0]) + margin
        y_min = np.min(X_Y0[:, 1]) - margin
        y_max = np.max(X_Y0[:, 1]) + margin
        if (x_max - x_min) > (y_max - y_min):
            h = (margin + x_max - x_min) / 2
            c = (y_max + y_min) / 2
            y_min = c - h
            y_max = c + h
        else:
            w = (margin + y_max - y_min) / 2
            c = (x_max + x_min) / 2
            x_min = c - w
            x_max = c + w
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        val0.set_data([],[])
        val1.set_data([],[])
        val2.set_data([],[])
        
        return val0, val1, val2, val3
    
    def update(frame):
        x0.append(frame[0])
        y0.append(frame[1])
        x1.append(frame[2])
        y1.append(frame[3])
        x2.append(frame[4])
        y2.append(frame[5])
        val0.set_data(x0, y0)
        val1.set_data(x1, y1)
        val2.set_data(x2, y2)
        
        cov_mat = frame[6:].reshape(2, -1)
        ellipse = error_ellipse(np.array([frame[4], frame[5]]), cov_mat)
        
        val3.angle = ellipse.angle
        val3.center = ellipse.center
        val3.width = ellipse.width
        val3.height = ellipse.height
        val3._alpha = ellipse._alpha
        
        return val0, val1, val2, val3
    
    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
    return anim

def build_traj_animation(enu_gt, enu_noise, enu_kf, title):

    fig = plt.figure()
    fig.set_size_inches(18, 18)
    ax = fig.add_subplot(1, 1, 1)
    print("Creating animation")

    ax.set_xlim((np.min(enu_gt[:,0])-10,np.max(enu_gt[:,0])+10))
    ax.set_ylim((np.min(enu_gt[:,1])-10,np.max(enu_gt[:,1])+10))
    ax.set_title(title)

    enu_gt_plt, = ax.plot([], [],'b',  animated=True, label='ENU GT')
    enu_noise_plt, = ax.plot([], [],'r:',  animated=True, label='ENU Noise')
    enu_kf_plt, = ax.plot([], [],'g--',  animated=True, label='ENU KF')
    plt.legend()
    
    def init():
        enu_gt_plt.set_data([], [])
        enu_noise_plt.set_data([], [])
        enu_kf_plt.set_data([], [])
        return  enu_gt_plt, enu_noise_plt, enu_kf_plt

    def animate(i):
        enu_gt_plt.set_data(enu_gt[:i,0], enu_gt[:i,1])
        enu_noise_plt.set_data(enu_noise[:i,0], enu_noise[:i,1])
        enu_kf_plt.set_data(enu_kf[:i,0], enu_kf[:i,1])
        return  enu_gt_plt, enu_noise_plt, enu_kf_plt

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=enu_gt.shape[0], interval=1, blit=True)

    return anim


def save_animation(ani, basedir, file_name):
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer)
    print("Animation saved")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    e1 = Ellipse(xy=(0.5, 0.5), width=0.5, height=0.2, angle=60, animated=True)
    e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100, animated=True)
    ax.add_patch(e1)
    ax.add_patch(e2)

    def init():
        return [e1, e2]

    def animate(i):
        e1.angle = e1.angle + 0.5
        e2.angle = e2.angle + 0.5
        return e1, e2

    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
    plt.show()