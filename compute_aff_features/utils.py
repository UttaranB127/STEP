import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return (180 / np.pi) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def distance_between(v1, v2):
    return np.linalg.norm(np.asarray(v1) - np.asarray(v2))

def heron(a,b,c):  
    s = (a + b + c) / 2   
    area = (s*(s-a) * (s-b)*(s-c)) ** 0.5        
    return area

def area_triangle(v1, v2, v3):  
    a = distance_between(v1, v2)
    b = distance_between(v1, v3)
    c = distance_between(v2, v3)
    A = heron(a,b,c)
    return A