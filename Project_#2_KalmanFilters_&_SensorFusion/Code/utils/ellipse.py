import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2


def draw_ellipse(ax, x, a, b, color):
    # Constants
    NPOINTS = 100 # point density or resolution

    # Compose point vector
    inc = 2*np.pi/NPOINTS
    ivec = np.arange(0, 2 * np.pi + inc, inc) # index vector
    N = ivec.shape[0]
    p = np.zeros((3, N))
    p[0, :] = a * np.cos(ivec) # 2 x n matrix which
    p[1, :] = b * np.sin(ivec) # hold ellipse points
    p[2, :] = np.ones(N)

    # Translate and rotate
    xo, yo, angle = x
    RT  = np.array([[np.cos(angle), -np.sin(angle), xo],
                    [np.sin(angle), np.cos(angle), yo],
                    [0, 0, 1]])
    p = RT.dot(p)

    # Plot
    ax_plot, = ax.plot(p[0,:], p[1,:], color, linewidth=2)
    return ax_plot


def draw_prob_ellipse(ax, x, C, alpha, color):
    # Calculate unscaled half axes
    sxx, syy, sxy = C[0, 0], C[1,1], C[0,1]
    a = np.sqrt(0.5*(sxx + syy + np.sqrt((sxx - syy)**2 + 4*sxy**2)) + 0j) # always greater
    b = np.sqrt(0.5*(sxx + syy - np.sqrt((sxx - syy)**2 + 4*sxy**2)) + 0j) # always smaller

    # Remove imaginary parts in case of neg. definite C
    a = a.real
    b = b.real

    # Scaling in order to reflect specified probability
    # a = a * np.sqrt(chi2invtable(alpha, 2))
    # b = b * np.sqrt(chi2invtable(alpha, 2))
    
    a = a * np.sqrt(chi2.ppf(alpha, df=2))
    b = b * np.sqrt(chi2.ppf(alpha, df=2))

    # Look where the greater half axis belongs to
    if sxx < syy:
        swap = a
        a = b
        b = swap

    # Calculate inclination (numerically stable)
    if sxx != syy:
        angle = 0.5 * np.arctan(2*sxy/(sxx-syy))
    elif sxy == 0:
        angle = 0 # angle doesn't matter 
    elif sxy > 0:
        angle =  np.pi/4
    elif sxy < 0:
        angle = -np.pi/4
    
    if len(x) < 3:
        x = np.hstack((x, angle))
    else:
        x[2] = angle

    # Draw ellipse
    return draw_ellipse(ax, x,a,b,color)

if __name__ == '__main__':
    # draw_ellipse([2,3, np.pi/6], 5, 9, 'green')
    draw_prob_ellipse([2,3, np.pi/4], np.array([[0.1, 0],[0, 0.1]]), 0.97, "blue")
    plt.show()
