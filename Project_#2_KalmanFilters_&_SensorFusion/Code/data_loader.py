import os
import pykitti
from utils import read_data


class DataLoader:
    def __init__(self, basedir, date, drive_num, dat_dir):
        self.data = pykitti.raw(basedir, date, drive_num)
        self.dat_dir = dat_dir
    
    def get_gps_imu(self):
        return self.data.oxts
    
    def get_timestamps(self):
        return self.data.timestamps
    
    def load_landmarks(self):
        # read landmarks for section 3
        world_path = os.path.join(self.dat_dir, "world.dat")
        return read_data.read_world(world_path)
    
    def load_sensor_data(self):
        # read sensor data for section 3
        sensor_data_path = os.path.join(self.dat_dir, "sensor_data.dat")
        return read_data.read_sensor_data(sensor_data_path)
    