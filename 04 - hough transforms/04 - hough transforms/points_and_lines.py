import numpy as np


def _ensure_shape(xy: np.ndarray) -> np.ndarray:
    if xy.ndim == 1:
        # Change things of shape (2,) to shape (1, 2) so that they conform to the (m,2) specification in all functions
        # below.
        xy = xy[np.newaxis, :]
    elif xy.ndim >= 3:
        raise ValueError(
            f"xy must have 1 or 2 dimensions, but has {xy.ndim} dimensions"
        )
    return xy


def cart2pol(xy: np.ndarray) -> np.ndarray:
    """Given m points in the Cartesian plane, return the polar coordinates of those points.

    :param xy: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array of shape (m, 2) where each row is (rho, theta)
    """
    xy = _ensure_shape(xy)
    ... # YOUR CODE HERE
    radius = np.hypot(xy[:, 0], xy[:, 1]).reshape([xy.shape[0],1])
    angle = np.arctan2(xy[:, 1],xy[:, 0]).reshape([xy.shape[0],1])
    
    output=np.concatenate((radius,angle),axis=1)
    # print(output.shape)
    return output


def pol2cart(rho_theta: np.ndarray) -> np.ndarray:
    """Given m points in the polar plane, return the Cartesian coordinates of those points.

    :param rho_theta: numpy array of shape (m, 2) where each row is (rho, theta)
    :return: numpy array of shape (m, 2) where each row is (x, y)
    """
    rho_theta = _ensure_shape(rho_theta)
    ... # YOUR CODE HERE
    rho = rho_theta[:, 0]
    theta = rho_theta[:, 1]
    
    # Convert polar coordinates to Cartesian coordinates
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    
    # Stack x and y to create the output array
    cartesian_coordinates = np.column_stack((x, y))
    return cartesian_coordinates



def unit_vector(v: np.ndarray):
    """Given m vectors in the Cartesian plane, return the unit vectors of those vectors.

    :param v: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array of shape (m, 2) where each row is (x', y')
    """
    v = _ensure_shape(v)
   
    # Calculate the magnitude (length) of each vector
    magnitude = np.linalg.norm(v, axis=1)
    
    # Ensure that we don't divide by zero
    non_zero_magnitudes = magnitude.copy()
    non_zero_magnitudes[magnitude == 0] = 1  # Replace zeros with 1 to avoid division by zero
    
    # Calculate the unit vectors by dividing each component by its magnitude
    unit_vectors = v / non_zero_magnitudes[:, np.newaxis]
    
    return unit_vectors


def point_on_line_closest_to_origin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Given two points in the Cartesian plane, return the point on the line between those points which is closest to
    the origin. This is the projection of the origin onto the line. Vectorized to handle m pairs of points at once.

    :param a: numpy array of shape (m, 2) where each row is (x, y)
    :param b: numpy array of shape (m, 2) where each row is (x, y)
    :return: numpy array p of shape (m, 2) such that p[i,:] is the projection of the origin onto the line a[i,:]--b[i,:]
    """
    a, b = _ensure_shape(a), _ensure_shape(b)
    ... # YOUR CODE HERE
 
    ab = b - a

    origin = np.zeros_like(a)
    projection = a + np.sum((origin - a) * ab, axis=1, keepdims=True) / np.sum(ab * ab, axis=1, keepdims=True) * ab
    
    return projection



    


def distance_point_to_normal_line(rho_theta: np.ndarray, xy: np.ndarray):
    """Given a line in normal form (rho, theta) and a point in the Cartesian plane (x, y), return the distance from the
    point to the line. Vectorized to handle m lines and points at once.

    :param rho_theta: numpy array of shape (m, 2) where each row is (rho, theta) and specificies a line in normal form
    :param xy: numpy array of shape (m, 2) where each row is (x, y) and specifies a point in the Cartesian plane
    :return: numpy array of shape (m,) where each element is the distance from xy[i] to the line defined by rho_theta[i]
    """
    rho_theta, xy = _ensure_shape(rho_theta), _ensure_shape(xy)
    ... # YOUR CODE HERE
    rho = rho_theta[:, 0]
    theta = rho_theta[:, 1]
    
    x = xy[:, 0]
    y = xy[:, 1]
    
    # Calculate the distance from each point to its corresponding line
    distance = np.abs(rho - x * np.cos(theta) - y * np.sin(theta))
    
    return distance

def get_all_lines_through_points(xy: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Given a set of m points xy in the Cartesian plane and an array of n angles, return a (m,n,2) array of lines in
    normal form. For example, if the output is stored in a variable 'lines', then lines[i,j,:] is (rho, theta) for the
    line through xy[i,:] at angle angles[j] (thus theta is equal to angles[j]).

    The definition of a line in normal form is

    .. math::
        \\rho = x \\cos(\\theta) + y \\sin(\\theta)

    This function essentially solves for rho in the above equation for each point in xy and each angle in angles.

    :param xy: numpy array of shape (m, 2) where each row is (x, y) and specifies a point in the Cartesian plane
    :param angles: numpy array of shape (n,) where each element is an angle in radians
    :return: numpy array of shape (m,n,2) where each [i,j,:] is (rho, theta) and specifies a line in normal form passing
        through xy[i,:] at angle angles[j]
    """
    xy = _ensure_shape(xy)
    ... # YOUR CODE HERE

    m, n = xy.shape[0], angles.shape[0]
    
    # Create empty array to store lines in normal form
    lines = np.zeros((m, n, 2))

    theta=angles.reshape(angles.shape[0],1)
    rho = xy[:,0] * np.cos(theta) + xy[:,1] * np.sin(theta)
    lines[:,:,0]=rho.T
    lines[:,:,1]=theta.T
    
    
    return lines






if __name__=='__main__':
    x=get_all_lines_through_points(
            np.array([[1, 0], [0, 1], [-1, -1]]),
            np.array([-np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi]),
        )
    print(x)