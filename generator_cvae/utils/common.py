import numpy as np
import torch
from torch.autograd import Variable


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(p1, p2, p3):
    """ Returns the angle in radians between vectors 'p1' - 'p2' and 'p3' - 'p2'::
    """
    u1 = unit_vector(p1 - p2)
    u2 = unit_vector(p3 - p2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))


def dist_between(v1, v2):
    """ Returns the l2-norm distance between vectors 'v1' and 'v2'::
    """
    return np.linalg.norm(v1-v2)


def area_of_triangle(v1, v2, v3):
    a = np.linalg.norm(v1 - v2)
    b = np.linalg.norm(v2 - v3)
    c = np.linalg.norm(v3 - v1)
    s = (a+b+c) / 2.
    return np.sqrt(s * (s-a) * (s-b) * (s-c))


def get_joints(g, s, t):
    root = g[s, t, 0, :]
    spine = g[s, t, 1, :] - root
    neck = g[s, t, 2, :] - root
    head = g[s, t, 3, :] - root
    rshoulder = g[s, t, 4, :] - root
    relbow = g[s, t, 5, :] - root
    rhand = g[s, t, 6, :] - root
    lshoulder = g[s, t, 7, :] - root
    lelbow = g[s, t, 8, :] - root
    lhand = g[s, t, 9, :] - root
    rhip = g[s, t, 10, :] - root
    rknee = g[s, t, 11, :] - root
    rfoot = g[s, t, 12, :] - root
    lhip = g[s, t, 13, :] - root
    lknee = g[s, t, 14, :] - root
    lfoot = g[s, t, 15, :] - root
    root = root - root
    return root, spine, neck, head, rshoulder, relbow, rhand,\
           lshoulder, lelbow, lhand, rhip, rknee, rfoot, lhip, lknee, lfoot


def get_velocity(pos_curr, pos_prev):
    vel = pos_curr - pos_prev
    return np.append(vel, np.linalg.norm(vel))


def get_acceleration(vel_curr, vel_prev):
    return vel_curr - vel_prev


def get_jerk(acc_curr, acc_prev):
    return np.linalg.norm(acc_curr - acc_prev)


def get_dynamics(pos_curr, pos_prev, vel_prev, acc_prev=None):
    vel_curr = get_velocity(pos_curr, pos_prev)
    acc_curr = get_acceleration(vel_curr[:-1], vel_prev)
    if acc_prev is None:
        return np.concatenate((vel_curr, acc_curr))
    jerk = get_jerk(acc_curr, acc_prev)
    return np.concatenate((vel_curr, acc_curr, [jerk]))


def get_affective_features(gaits):
    # 0: root, 1: spine, 2: neck, 3: head
    # 4: rshoulder, 5: relbow, 6: rhand
    # 7: lshoulder, 8: lelbow, 9: lhand
    # 10: rhip, 11: rknee, 12: rfoot
    # 13: lhip, 14: lknee, 15: lfoot

    num_samples = gaits.shape[0]
    num_tsteps = gaits.shape[1]
    num_features = 175
    up_vector = np.array([0, 1, 0])
    affective_features = np.zeros((num_samples, num_tsteps, num_features))
    Y = np.array(get_joints(gaits, 0, 0)).transpose()
    for sidx in range(num_samples):
        X = np.array(get_joints(gaits, sidx, 0)).transpose()
        R, c, t = get_transformation(X, Y)
        for tidx in range(num_tsteps):
            fidx = 0
            Xtx = np.array(get_joints(gaits, sidx, tidx)).transpose()
            affective_features[sidx, tidx, fidx:fidx+48] = (np.dot(c*R, Xtx) + np.tile(
                np.reshape(t, (t.shape[0], 1)), (1, Xtx.shape[1]))).transpose().flatten()
            fidx += 48
            # within frame
            _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15 = get_joints(gaits, sidx, tidx)
            affective_features[sidx, tidx, fidx] = angle_between_points(_7, _2, _4)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_2, _4, _7)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_4, _7, _2)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_vectors(_3-_0, up_vector)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_6, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_9, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_6, _4)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_9, _7)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_5, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_8, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_9, _2, _6)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_7, _2, _4)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_9, _0, _6)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_8, _2, _5)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_2, _4, _5)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_2, _7, _8)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_4, _5, _6)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_7, _8, _9)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_2, _1, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_3, _2, _1)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_12, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_15, _0)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_15, _2, _12)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_13, _2, _10)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_15, _0, _12)
            fidx += 1
            affective_features[sidx, tidx, fidx] = area_of_triangle(_14, _2, _11)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_0, _10, _11)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_0, _13, _14)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_10, _11, _12)
            fidx += 1
            affective_features[sidx, tidx, fidx] = angle_between_points(_13, _14, _15)
            fidx += 1
            affective_features[sidx, tidx, fidx] = dist_between(_12, _15)
            fidx += 1

            # between frames
            if tidx > 0:
                _0_1, _1_1, _2_1, _3_1, _4_1, _5_1, _6_1, _7_1,\
                    _8_1, _9_1, _10_1, _11_1, _12_1, _13_1, _14_1, _15_1 = get_joints(gaits, sidx, tidx-1)
                affective_features[sidx, tidx, fidx:fidx+8] =\
                    get_dynamics(_6, _6_1, affective_features[sidx, tidx - 1, fidx:fidx+3],
                                 affective_features[sidx, tidx-1, fidx+4:fidx+7])
                fidx += 8
                affective_features[sidx, tidx, fidx:fidx+8] =\
                    get_dynamics(_9, _9_1, affective_features[sidx, tidx - 1, fidx:fidx+3],
                                 affective_features[sidx, tidx-1, fidx+4:fidx+7])
                fidx += 8
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_5, _5_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_8, _8_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_4, _4_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_7, _7_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+8] = \
                    get_dynamics(_12, _12_1, affective_features[sidx, tidx - 1, fidx:fidx+3],
                                 affective_features[sidx, tidx - 1, fidx+4:fidx+7])
                fidx += 8
                affective_features[sidx, tidx, fidx:fidx+8] = \
                    get_dynamics(_15, _15_1, affective_features[sidx, tidx - 1, fidx:fidx+3],
                                 affective_features[sidx, tidx - 1, fidx+4:fidx+7])
                fidx += 8
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_11, _11_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_14, _14_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_10, _10_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+7] =\
                    get_dynamics(_13, _13_1, affective_features[sidx, tidx - 1, fidx:fidx+3])
                fidx += 7
                affective_features[sidx, tidx, fidx:fidx+8] =\
                    get_dynamics(_3, _3_1, affective_features[sidx, tidx - 1, fidx:fidx+3],
                                 affective_features[sidx, tidx - 1, fidx+4:fidx+7])
                fidx += 8

    return affective_features


def get_transformation(X, Y):
    """

    Args:
        X: k x n source shape
        Y: k x n destination shape such that Y[:, i] is the correspondence of X[:, i]

    Returns: rotation R, scaling c, translation t such that ||Y - (cRX+t)||_2 is minimized.

    """
    """
    Copyright: Carlo Nicolini, 2013
    Code adapted from the Mark Paskin Matlab version
    from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
    """

    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc*Xc, 0))
    sy = np.mean(np.sum(Yc*Yc, 0))

    M = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(M, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    #print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(M)
    d = np.linalg.det(M)
    S = np.eye(m)
    if r > (m - 1):
        if np.linalg.det(M) < 0:
            S[m, m] = -1
    elif r == m - 1:
        if np.linalg.det(U) * np.linalg.det(V) < 0:
            S[m, m] = -1
    else:
        R = np.eye(2)
        c = 1
        t = np.zeros(2)
        return R, c, t

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)

    return R, c, t
