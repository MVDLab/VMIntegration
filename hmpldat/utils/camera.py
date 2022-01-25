"""Methods for relating focal distance and camera params to pixel size

"""

import numpy as np

def calc_pixel_size(d, theta, phi, u, v):
    """

    Assuming camera is calibrated

    Args:
        d: working distance to object in physical units (millimeter, foot, fathom, don't mater)
        theta:  horizontal field of view (in degrees)
        phi: vertical FoV (in degrees)
        h: number of horizontal pixels
        v: number of vertical pixels

    Returns:
        w, h in the physical units of focal length argument

    """

    theta = np.radians(theta / 2)
    phi = np.radians(phi / 2)
    
    if (v % 2 != 0) or (u % 2 != 0):
        raise ValueError("odd number of vertical or horizontal pixels")
    
    u = u/2
    v = v/2

    w = (d * np.tan(theta)) / u
    h = (d * np.tan(phi)) / v
    
    return w, h

