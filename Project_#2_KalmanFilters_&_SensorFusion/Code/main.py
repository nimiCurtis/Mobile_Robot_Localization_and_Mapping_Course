# Mapping and perception for autonomous robot , semester B 2022/23
# Roy orfaig & Ben-Zion Bobrovsky
# Project 2 - Robot localization and SLAM!
# Kalman filter, Extended Kalman Filter and EKF-SLAM. 

import os
from data_loader import DataLoader
from project_questions import ProjectQuestions


if __name__ == "__main__":

    #NOTE: change the path below according to your specific setup!! 
    basedir = "../../../../kitti_data" 
    date =  '2011_09_30'  
    drive =    '0020' 

    dat_dir = os.path.join(basedir,"Ex2_data")
    
    dataset = DataLoader(basedir, date, drive, dat_dir)
    
    project = ProjectQuestions(dataset)
    project.run()
