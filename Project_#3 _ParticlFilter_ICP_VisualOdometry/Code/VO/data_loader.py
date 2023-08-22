import os
import numpy as np
import cv2
from camera import Camera


class DataLoader:
    def __init__(self, vo_data):
	# read your data
        # read your data
        poses_path = os.path.join(vo_data['dir'], "poses", f"{vo_data['sequence']:02}.txt")
        calib_path = os.path.join(vo_data['dir'], "sequences", f"{vo_data['sequence']:02}", "calib.txt")
        times_path = os.path.join(vo_data['dir'], "sequences", f"{vo_data['sequence']:02}", "times.txt")
        img_dir = os.path.join(vo_data['dir'], "sequences", f"{vo_data['sequence']:02}", "image_0")

        assert os.path.isfile(poses_path), "poses file does not exists"
        assert os.path.isfile(calib_path), "calib file does not exists"
        assert os.path.isfile(times_path), "times file does not exists"
        assert os.path.isdir(img_dir), "images dir does not exists"
        
        P0, Tr = self._load_calib(calib_path)
        self.cam = Camera(P0, Tr)
        self.N = self._get_number_of_frames(times_path)
        self.img_dir = img_dir
        self.poses_path = poses_path
    
    @staticmethod
    def line2mat(line):
        assert type(line) is str
        return np.array(list(map(float, line.replace('\n','').split(' ')))).reshape(-1, 4)
    
    def _load_calib(self, calib_path):
        with open(calib_path, "r") as f:
            lines = f.readlines()
        P0 = DataLoader.line2mat(lines[0][4:])[:, :-1]
        Tr = DataLoader.line2mat(lines[4][4:])
        return P0, Tr
    
    def _get_number_of_frames(self, times_path):
        with open(times_path, "r") as f:
            lines = f.readlines()
        
        time2float = lambda val: float(val.replace('\n',''))
        times = list(map(time2float, list(filter(None, lines))))
        return len(times)
    
    @property
    def images(self, mode=cv2.IMREAD_GRAYSCALE):
        root, _, files = next(os.walk(self.img_dir))
        files_path = list(map(lambda val: os.path.join(root, val), files))
        
        for file in files_path:
            yield cv2.imread(file, mode)
    
    @property
    def gt_poses(self):
        with open(self.poses_path, 'r') as poses_file:
            lines = poses_file.readlines()
        
        for line in lines:
            yield DataLoader.line2mat(line)
    

    def make_mp4(self):
        import cv2
        import os

        image_folder = self.img_dir
        video_name = self.img_dir + '/video.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 20, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
