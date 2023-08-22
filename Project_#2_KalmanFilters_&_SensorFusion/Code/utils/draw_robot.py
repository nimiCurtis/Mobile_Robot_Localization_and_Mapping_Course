import numpy as np
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt


#DRAWROBOT Draw robot.
#   DRAWROBOT(X,COLOR) draws a robot at pose X = [x y theta] such
#   that the robot reference frame is attached to the center of
#   the wheelbase with the x-axis looking forward. COLOR is a
#   [r g b]-vector or a color string such as 'r' or 'g'.
#
#   DRAWROBOT(X,COLOR,TYPE) draws a robot of type TYPE. Five
#   different models are implemented:
#      TYPE = 0 draws only a cross with orientation theta
#      TYPE = 1 is a differential drive robot without contour
#      TYPE = 2 is a differential drive robot with round shape
#      TYPE = 3 is a round shaped robot with a line at theta
#      TYPE = 4 is a differential drive robot with rectangular shape
#      TYPE = 5 is a rectangular shaped robot with a line at theta
#
#   DRAWROBOT(X,COLOR,TYPE,W,L) draws a robot of type TYPE with
#   width W and length L in [m].
#
#   H = DRAWROBOT(...) returns a column vector of handles to all
#   graphic objects of the robot drawing. Remember that not all
#   graphic properties apply to all types of graphic objects. Use
#   FINDOBJ to find and access the individual objects.
#
#   See also DRAWRECT, DRAWARROW, FINDOBJ, PLOT.

# v.1.0, 16.06.03, Kai Arras, ASL-EPFL
# v.1.1, 12.10.03, Kai Arras, ASL-EPFL: uses drawrect
# v.1.2, 03.12.03, Kai Arras, CAS-KTH : types implemented


def draw_robot(*varargin):

    # Constants
    DEFT = 2;            # default robot type
    DEFB = 0.4;          # default robot width in [m], defines y-dir. of {R}
    WT   = 0.03;         # wheel thickness in [m]
    DEFL = DEFB+0.2;     # default robot length in [m]
    WD   = 0.2;          # wheel diameter in [m]
    RR   = WT/2;         # wheel roundness radius in [m]
    RRR  = 0.04;         # roundness radius for rectangular robots in [m]
    HL   = 0.09;         # arrow head length in [m]
    CS   = 0.1;          # cross size in [m], showing the {R} origin

    # Input argument check
    inputerr = 0
    nargin = len(varargin)
    # if(nargin == 2):
    #     xvec  = varargin[0]
    #     color = varargin[1]
    #     type_  = DEFT
    #     B     = DEFB
    #     L     = DEFL
    # elif(nargin == 3):
    #     xvec  = varargin[0]
    #     color = varargin[1]
    #     type_  = varargin[2]
    #     B     = DEFB
    #     L     = DEFL
    if(nargin == 6):
        ax = varargin[0]
        
        xvec = varargin[1][:3]
        # xmat = varargin[1]
        # xvec = xmat[-1, 0:3]
        
        color = varargin[2]
        type_  = varargin[3]
        B     = varargin[4]
        L     = varargin[5]
    else:
        inputerr = 1

    # Main switch statement
    if inputerr == 0:
        x, y, theta = xvec[0], xvec[1], xvec[2]
        T = np.array([x, y])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        # x, y, theta = xmat[:, 0], xmat[:, 1], xmat[:, 2]
        # T = np.array([x, y]).T
        # R = np.array([np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]).T
            
        if type_ == 3:
            # Draw circular contour
            radius = (B+WT)/2
            h1 = draw_ellipse(ax, xvec, radius, radius, color)
            # Draw line with orientation theta with length radius
            p = R.dot(np.array([radius, 0])) + T
            # p = np.array([radius * R[:,0], radius * R[:,1]]).T + T
            
            # h2, = ax.plot(np.array([T[:,0], p[:,0]]), np.array([T[:,1], p[:,1]]), c=color, linewidth=2) # TODO: fix
            h2, = ax.plot(np.array([T[0], p[0]]), np.array([T[1], p[1]]), c=color, linewidth=2) # TODO: fix
            # h = cat(1,h1,h2)
            return h1, h2
            
        
            
        else:
            raise RuntimeError('drawrobot: Unsupported robot type')
        
    else:
        raise RuntimeError('drawrobot: Wrong number of input arguments')
