import os
#from visual_odometry_alon import VisualOdometry
from visual_odometry import VisualOdometry
from data_loader import DataLoader


class ProjectQuestions:
    def __init__(self,vo_data):
        assert type(vo_data) is dict, "vo_data should be a dictionary"
        assert all([val in list(vo_data.keys()) for val in ['sequence', 'dir']]), "vo_data must contain keys: ['sequence', 'dir']"
        assert type(vo_data['sequence']) is int and (0 <= vo_data['sequence'] <= 10), "sequence must be an integer value between 0-10"
        assert type(vo_data['dir']) is str and os.path.isdir(vo_data['dir']), "dir should be a directory"
        self.vo_data = vo_data

    
    
    def Q3(self):
        vo_data = DataLoader(self.vo_data)
        vo = VisualOdometry(vo_data,self.vo_data['results'])
        vo.calc_trajectory()
        
    
    def run(self):
        self.Q3()
    