"""Basic math methods

"""

from time import time_ns
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean

# TODO: compare speed scipy.spatial.distance.euclidean: use faster
def euclidean_distance(u, v, squared=False):
    """what is the distance between 2 points?

    .. math::

        \\sqrt{\\sum_{i}{(u_{i} - v_{i})^2}}

    Args:
        u : 1D array_like [x,y,z]
        v : 1D array_like [x,y,z]
        squared : default=False; return squared distance (do not take square root)

    Returns:
        float

    Note:
        handles any number of dimensions, but u and v must have the same dimensionality

        columns == attributes ['X', 'Y', 'Z', 'Q', etc.]

    """

    if len(u) != len(v):
        raise ValueError(f"inputs are different shapes u=[{u}] v=[{v}]")

    squared_dist = sum((p - q) ** 2 for p, q in zip(u, v))

    return squared_dist if squared else squared_dist ** 0.5 


def vectorized_euclidean_dist(u, v, squared=False):
    """
    Args:
        u: 2D array of x,y,z positions(column) for each instance(row)
        v: 2D array of x,y,z positions(column) for each instance(row)
        squared : default=False; return squared distance (do not take square root)

    Returns:
        1D array with distance for each instance

    """

    x1, y1, z1 = u
    x2, y2, z2 = v

    squared_dist = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

    return squared_dist if squared else dist ** 0.5


def unit_vector(u, v):
    """ 
    u - v
    
    """

    if len(u) != len(v):
        raise ValueError(f"inputs are different shapes u=[{u}] v=[{v}]")

    vector = np.array([(p - q) for p, q in zip(u, v)])
    magnitude = np.linalg.norm(vector)

    return vector / magnitude


def angle_between_vectors(a, b):
    """

    .. math::

        \\theta = arccos( \\frac{a dot b}{mag(a)*mag(b)})

    """

    # vectors must be non-zero
    if not any(a) or not any(b):
        return np.NaN

    # todo: CHACK SHAPED

    theta = np.arccos(
        sum(u * v for u, v in zip(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

    return theta  # in radians


def quadratic_formula(a, b, c):
    """just like the song

    .. math::
    
        ax^2 + bx + c = 0
        
        solns = \\frac{-b \\pm \\sqrt{b^2 -4ac}}{2a}

    Returns:
        solutions (only real):
        * NaN when discriminant is negative
        * single soln when discriminate == 0
        * two solns otherwise
       
    """

    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0 or np.isnan(discriminant):
        return np.NaN
    elif discriminant == 0:
        return -b / (2 * a)
    else:
        return (
            (-b + discriminant ** 0.5) / (2 * a),
            (-b - discriminant ** 0.5) / (2 * a),
        )


# TODO: not used, remove this?
def slope_2d(p2, p1):
    """ returns rise and run for each instance in the xz plane

    p2 - p1

    ::
        -z
        ^
        |
        |
        +------> +x

    """

    return pd.DataFrame({"rise": p2.Z - p1.Z, "run": p2.X - p1.X})


def point_plane_distance(plane, point):
    """
    calculate shortest distance between a point and a plane

    Args:
        plane : (array_like) [[x,y,z],[x,y,z],[x,y,z]] three points describing plane
        point : (array_like) [x,y,z] a point

    Returns: 
        float, distance from point to plane

    Notes:
        .. https://mathworld.wolfram.com/Point-PlaneDistance.html
        sign indicate direction?? add abs()??

    """

    p_a, p_b, p_c = plane

    ab = p_b - p_a
    ac = p_c - p_a

    # find vector normal to plane
    n = np.cross(ab, ac)

    # divide each element bt magnitude to create unmit vector
    u = unit_vector(n, [0, 0, 0])

    # project vector p1->point to normal unit vector
    return np.dot(u, point - p_c)


def normal_vector_from_3points(p1, p2, p3):
    """ 
    
    Args:
        p1 : (array_like) [x, y, z] a point
        p2 : (array_like) [x, y, z] a point
        p3 : (array_like) [x, y, z] a point

    Returns: 
        A normal vector describing plane

    """

    return np.cross((p3 - p1), (p2 - p1))


def bounded_by_triangle(p, a, b, c):
    """ is p inside a, b, & c 
    
    Barycentric Technique
    """

    # Compute vectors
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # // Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # // Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # // Check if point is in triangle

    return (u >= 0) and (v >= 0) and (u + v < 1)


def on_same_plane(p1, p2, p3, p4):
    """ check that these four points are on the same plane 
    
        returns:
            float ~= 0 if true

    """

    arr = np.array([p1, p2, p3, p4])
    ones = np.ones((4, 1))

    m = np.hstack((arr, ones))

    return np.linalg.det(m)


def on_same_plane_2(p1, p2, p3, p4):
    """ also works, but slower"""
    a = p2 - p1
    b = p3 - p1
    c = p4 - p1

    return np.dot(c, np.cross(a, b))


def rigid_transform_3D(a, b, t=None):
    """    
    Takes a set of points defining configuration A and finds the rigid transform taking them to a set of points defining configuration B.
    Configuration A and configuration B must have the same number of points.

    [[x1, x2, x3, x4, ..., xn],
     [y1, y2, y3, y4, ..., yn],
     [z1, z2, z3, z4, ..., zn]]

    Args:
        a: An array of n cordinate points describing configuration A.
        b: An array of n cordinate points describing configuration B.
        t: (3,) known translation, if given, translation is not calculated

    Returns:
        r: Rotation matrix (3,3)
        t: Translation vector (3,)
    
    Notes:
        .. https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
        Modified to use ndarray (NOT numpy matrix object):

    """

    if a.shape != b.shape:
        raise ValueError("a and b must be the same shape")

    num_rows, num_cols = a.shape

    if num_rows != 3:
        raise Exception("an array is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # if a translation is provided, translate b to origin
    if t is not None:
        b = b - t.reshape(3,1)

    # find mean, row wise => centroid
    centroid_a = np.mean(a, axis=1)
    centroid_b = np.mean(b, axis=1)

    # translate origin to centroid for each group of points (reshape 1D vector into column vector)
    am = a - centroid_a.reshape(3,1)
    bm = b - centroid_b.reshape(3,1)

    # find rotation
    h = am @ bm.T
    u, _s, vh = np.linalg.svd(h)
    r = vh.T @ u.T

    # handle special reflection case
    if np.linalg.det(r) < 0:
        vh[2, :] *= -1
        r = vh.T @ u.T

    # if translation is not provided, calculate
    if t is None:
        t = -r @ centroid_a + centroid_b

    return r, t


def point_line_distance(x0, x1, x2):
    """

    Args:
        x0 : find the distance from this point[x,y,z] to the line
        x1 : this point[x,y,z] defines the line
        x2 : this point[x,y,z] defines a line

    Returns:


    Note:
        .. https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    """

    return abs(np.cross(x0 - x1, x0 - x2)) / abs(x2 - x1)


def ray_cylinder_intersect(vector_origin, passing_thru, radius=2490):
    """Method to find vector screen intersect

    this cylinder exists around the y-axis
    x^2 + z^2 = radius^2

    Args:
        vector_origin: a point [x,y,z]
        passing_thru: a point [x,y,z]

    Notes:
        .. https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html#eqn:rectray

    """

    x_o, _y_o, z_o = vector_origin

    # define a ray direction (vector)
    d = passing_thru - vector_origin
    x_d, _y_d, z_d = d
    # print(x_d, _y_d, z_d)

    a = x_d ** 2 + z_d ** 2
    b = 2 * (x_o * x_d + z_o * z_d)
    c = x_o ** 2 + z_o ** 2 - radius ** 2

    t = quadratic_formula(a, b, c)
    # print(t)

    # solution is always the smallest non-negative!
    # since participant is inside the cylinder and looking forward (you don't have eyes in the back of your head)
    # the quadratic should return a positive (forward) and negative (backwards) solution.
    if isinstance(t, tuple):
        t = max(t)

    return vector_origin + t * d


def ray_cylinder_intersect_test(vector_origin, passing_thru, radius=2490):
    """Method to find vector screen intersect

    TEST IF USING PASSING THROUGH AS EYE POINT CHANGES SOLN

    this cylinder exists around the y-axis
    x^2 + z^2 = radius^2

    Args:
        vector_origin: a point [x,y,z]
        passing_thru: a point [x,y,z]

    Notes:
        https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html#eqn:rectray

    """

    x_o, _y_o, z_o = passing_thru

    # define a ray direction (vector)
    d = passing_thru - vector_origin
    x_d, _y_d, z_d = d
    # print(x_d, _y_d, z_d)

    a = x_d ** 2 + z_d ** 2
    b = 2 * (x_o * x_d + z_o * z_d)
    c = x_o ** 2 + z_o ** 2 - radius ** 2

    t = quadratic_formula(a, b, c)
    # print(t)

    # solution is always the smallest non-negative!
    if isinstance(t, tuple):
        t = max(t)

    return passing_thru + t * d


def ray_plane_intersect(vector_origin, passing_thru, norm_vector, point_on_plane):
    """
    
    Args:
        vector_origin: point [x,y,z] defines ray
        passing_thru: point [x,y,z] defines ray
        norm_vector: non-zero normal vector defining plane (typically: camera - frame_center)
        point_on_plane: [x,y,z] just like it sounds
        
    Notes:
        https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html#SECTION00023500000000000000

    """

    # d
    d = passing_thru - vector_origin
    t = norm_vector @ (point_on_plane - vector_origin) / (norm_vector @ d)

    return vector_origin + t * d


def ray_plane_intersect_test(vector_origin, passing_thru, norm_vector, point_on_plane):
    """
    
    TEST IF USING PASSING THROUGH AS EYE POINT CHANGES SOLN


    Args:
        vector_origin: point [x,y,z] defines ray
        passing_thru: point [x,y,z] defines ray
        norm_vector: non-zero normal vector defining plane (typically: camera - frame_center)
        point_on_plane: [x,y,z] just like it sounds
        
    Notes:
        https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html#SECTION00023500000000000000

    """

    d = passing_thru - vector_origin
    t = norm_vector @ (point_on_plane - passing_thru) / (norm_vector @ d)

    return passing_thru + t * d


from scipy.spatial.distance import cdist
from pprint import pprint
from time import time_ns

if __name__ == "__main__":
    # x = np.random.rand(10,3)
    # y = np.random.rand(10,3)

    # print(sum((u - v)**2 for u,v in zip(x[0], y[0]))**0.5)

    # pprint(x)
    # pprint(y)

    # r = []
    # i=0
    # t = time_ns()
    # for rx, ry in zip(x, y):
    #     r.append(euclidean_distance(rx, ry))
    #     i+=1

    # avg = np.mean(r)
    # print(time_ns() - t)
    # print(avg)
    # print(r)

    # t = time_ns()
    # # use this for euclidean distance on arrays
    # res = np.mean(np.diagonal(cdist(x, y, 'euclidean')))
    # print(time_ns() - t)
    # print(res)

    # camera = np.array([ -18.92569708, 1658.2985533 , -142.14010816])
    # vector_origin = np.array([ -14.14418458, 1655.84771774, -132.61957549])
    camera = np.array([15.35982161, 1659.66176068, -161.84452562])
    vector_origin = np.array([15.68271436, 1657.27412637, -151.62469477])

    ray_cylinder_intersect(vector_origin, camera)
