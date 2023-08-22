
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def error_ellipse(position, sigma):

    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse



  #      ellipse = error_ellipse(landmark['mu'], landmark['sigma'])
  #      plt.gca().add_artist(ellipse)
