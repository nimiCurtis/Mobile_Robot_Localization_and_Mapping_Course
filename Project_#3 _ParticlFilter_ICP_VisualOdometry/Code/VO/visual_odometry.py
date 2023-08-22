import numpy as np
import cv2
from data_loader import DataLoader
from camera import Camera
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import moviepy.video.io.VideoFileClip as mp

class VisualOdometry:
    def __init__(self, vo_data, results_dir):
        """
        Initialize the VO class with the loaded data vo_data
        lastly, initialize the neutral rotation and translation matrices
        """
        self.vo_data = vo_data
        self.results_dir = results_dir

        # initial camera pose
        self.camera_rotation = vo_data.cam.extrinsics[:,:3]
        self.camera_translation =  vo_data.cam.extrinsics[:,3:]

    def calc_trajectory(self):
        """
        apply the visual odometry algorithm
        """
        gt_trajectory = np.array([]).reshape(0, 2)
        measured_trajectory = np.array([]).reshape(0, 2)
        key_points_history = []
        xz_error_arr = []

        #Initialize the SIFT detector & keypoints matcher
        feature_detector = cv2.SIFT_create() 
        feature_matcher = cv2.BFMatcher() 
        scale_factor = 1  # Initialize scale factor to 1.0
        

        idx_frames_for_save = range(0,4500,250)

        dest_dir = self.results_dir        
        fig = plt.figure(figsize=[16, 12])
        grid = plt.GridSpec(12, 17, hspace=0.2, wspace=0.2)
        
        
        ax_image = fig.add_subplot(grid[:5, :], title="Scene Image")
        ax_error_plot = fig.add_subplot(grid[6:, :8], title="Euclidean Distance Error", xlabel="Frame number", ylabel="Error [m]")
        ax_trajectory = fig.add_subplot(grid[6:, 9:], title="Trajectory", xlabel="X [m]", ylabel="Y [m]")

        Frames = []
        
        prev_img = None
        prev_gt_pose = None
        i = 0

        for curr_img, curr_gt_pose in zip(self.vo_data.images, self.vo_data.gt_poses):                
            if prev_img is None:
                prev_img = curr_img
                prev_gt_pose = curr_gt_pose
                continue


            #*********** 
            #TODO
            #***********

            curr_kp, curr_desc = feature_detector.detectAndCompute(curr_img, None)
            prev_kp, prev_desc = feature_detector.detectAndCompute(prev_img, None)

            # Match keypoints between the current and previous frames
            matches = feature_matcher.knnMatch(curr_desc,prev_desc , k=2)

            # Apply ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Extract matched keypoints
            curr_matched_pts = np.float32([curr_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            prev_matched_pts = np.float32([prev_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate the essential matrix using RANSAC
            essential_matrix, _ = cv2.findEssentialMat( prev_matched_pts, curr_matched_pts,self.vo_data.cam.intrinsics) #, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Recover the rotation and translation from the essential matrix
            _, R, t, _ = cv2.recoverPose(essential_matrix,  prev_matched_pts,curr_matched_pts, self.vo_data.cam.intrinsics)
            
            gt_scale = np.linalg.norm(curr_gt_pose - prev_gt_pose)
            vo_scale = np.linalg.norm(t)

            scale_factor = gt_scale / vo_scale
            # Update the camera pose
            self.camera_translation = self.camera_translation + scale_factor*(self.camera_rotation @ t)
            self.camera_rotation = self.camera_rotation @ R  

            gt_trajectory = np.concatenate((gt_trajectory, np.array([[curr_gt_pose[0, 3], curr_gt_pose[2, 3]]])), axis=0)  # Ground Truth
            measured_trajectory = np.concatenate((measured_trajectory, np.array([[float(self.camera_translation[2]), float(self.camera_translation[1])]])), axis=0)

            # Save keypoints for visualization (optional)
            key_points_history.append(curr_matched_pts.reshape(-1,2))

            # Calculate error in (x, y) Euclidean distance
            xz_error = np.linalg.norm(gt_trajectory[-1] - measured_trajectory[-1])
            xz_error_arr.append(xz_error)
            
            if i in idx_frames_for_save:
                print(f"iter: {i} , error: {xz_error}")
                print("save image")

            # Set the current image and ground truth pose as previous for the next iteration
            prev_img = curr_img
            prev_gt_pose = curr_gt_pose
            # Call the visualization function
            frame_graph = self.visualization(gt_trajectory, measured_trajectory, prev_matched_pts.reshape(-1,2), curr_matched_pts.reshape(-1,2), xz_error_arr, curr_img,
                                            ax_image, ax_error_plot, ax_trajectory, i, idx_frames_for_save, dest_dir)

            Frames.append(frame_graph)

            i+=1

        #show final plot
        print("last frame..")
        print(f"iter: {i} , error: {xz_error}")
        frame_graph = self.visualization(gt_trajectory, measured_trajectory, prev_matched_pts.reshape(-1,2), curr_matched_pts.reshape(-1,2), xz_error_arr, curr_img,
                                            ax_image, ax_error_plot, ax_trajectory, i-1, [i-1], dest_dir)
        Frames.append(frame_graph)
        ani = animation.ArtistAnimation(fig,Frames, interval=200)
        self.save_animation(ani, dest_dir, "Visual_Odometry_Animation")

        return gt_trajectory, measured_trajectory, key_points_history



        

    #Helpfull reference
    @staticmethod
    def visualization(GT_location,VO_location,prev_points,curr_points,xz_error_arr,curr_image,ax_image, ax_error_plot, ax_trajectory, frame_idx,idx_frames_for_save,dest_dir):
        """
        plot the graphes of the VO include: image, GT and estimated trajectory, features.
        :param GT_location: GT location
        :param VO_location: VO estimated location
        :param prev_points: KeyPoints from the previous frame.
        :param curr_points: match KeyPoints in the current frame of the previos frame.
        :param xz_error_arr: euclidian distance error in (x,y)
        :param curr_image: current image
        :param ax_image: Axis object for the image
        :param ax_error_plot: Axis object for the error plot
        :param ax_trajectory: Axis object for the trajectory plot
        :param frame_idx: frame index
        :param idx_frames_for_save: the indexes of the frames we want to save their graphs.
        :param dest_dir: the directory name for saving the graphs and animations to.
        :return: the frame graph.
        """
        Frame=[]
        plot_0=ax_image.imshow(curr_image,cmap='gray')
        Frame.append(plot_0)
        plot_1=ax_image.scatter(curr_points[:,0],curr_points[:,1],s=2,linewidths=0.5,edgecolors="b",marker="o")
        Frame.append(plot_1)
        plot_2=ax_image.scatter(prev_points[:, 0], prev_points[:, 1], s=2,linewidths=0.5, edgecolors="g",marker="P")
        Frame.append(plot_2)
        plot_3,=ax_trajectory.plot(VO_location[:,0],VO_location[:,1],c="r")
        Frame.append(plot_3)
        plot_4,=ax_trajectory.plot(GT_location[:, 0], GT_location[:, 1],"--b")
        Frame.append(plot_4)
        plot_5, = ax_error_plot.plot(xz_error_arr,c="orange")
        Frame.append(plot_5)
        if frame_idx == 1:
            ax_image.legend(["current key points", "Previous key points"],loc="upper right")
            ax_trajectory.legend(["VO-Estimated with scale", "GT"],loc="upper right")
            ax_trajectory.grid()
            ax_error_plot.grid()

        if frame_idx in idx_frames_for_save:
            fig_2 = plt.figure(figsize=[16, 12])
            grid = plt.GridSpec(12, 17, hspace=0.2, wspace=0.2)
            ax_image_2 = fig_2.add_subplot(grid[:5, :], title="Scene Image,Frame: {}".format(frame_idx))
            ax_error_plot_2 = fig_2.add_subplot(grid[6:, :8], title="Euclidean Distance Error", xlabel="Frame number",ylabel="Error[m]")
            ax_trajectory_2 = fig_2.add_subplot(grid[6:, 9:], title="Trajectory", xlabel="X[m]", ylabel="Y[m]",xlim=(-50, 750), ylim=(-100, 1000))
            ax_image_2.axis('off')
            ax_image_2.imshow(curr_image, cmap='gray')
            ax_image_2.scatter(curr_points[:, 0], curr_points[:, 1], s=2, linewidths=0.5, edgecolors="b", marker="o")
            ax_image_2.scatter(prev_points[:, 0], prev_points[:, 1], s=2, linewidths=0.5, edgecolors="g", marker="P")
            ax_trajectory_2.plot(VO_location[:, 0], VO_location[:, 1], c="r")
            ax_trajectory_2.plot(GT_location[:, 0], GT_location[:, 1], "--b")
            ax_error_plot_2.plot(xz_error_arr, c="orange")
            ax_image_2.legend(["current key points", "Previous key points"], loc="upper right")
            ax_trajectory_2.legend(["VO-Estimated with scale", "GT"], loc="upper right")
            ax_trajectory_2.grid()
            ax_error_plot_2.grid()
            plt.savefig(dest_dir + "/Visual Odometry Frame #{}".format(frame_idx))


        return Frame

    @staticmethod
    def save_animation(ani, basedir, file_name):
        """
        save animation function
        :param ani: animation object
        :param basedir: the parent dir of the animation dir.
        :param file_name: the animation name
        :return: None
        """
        print("Saving animation")
        if not os.path.exists(basedir + "/Animation videos"):
            os.makedirs(basedir + "/Animation videos")
        gif_file_path = os.path.join(basedir + "/Animation videos", f'{file_name}.gif')
        mp4_file_path = os.path.join(basedir + "/Animation videos", f'{file_name}.mp4')

        writergif = animation.PillowWriter(fps=10)
        ani.save(gif_file_path, writer=writergif)
        
        clip = mp.VideoFileClip(gif_file_path)
        clip.write_videofile(mp4_file_path)
        os.remove(gif_file_path)
        print("Animation saved")