import numpy as np

# from utils import angle_between
# from utils import distance_between
# from utils import area_triangle
from compute_aff_features.utils import angle_between
from compute_aff_features.utils import distance_between
from compute_aff_features.utils import area_triangle


# Volume of the bounding box
def compute_feature0_per_frame(frame):
    min_x = float('inf')
    min_y = float('inf')
    min_z = float('inf')

    max_x = float('-inf')
    max_y = float('-inf')
    max_z = float('-inf')
    for i in range(16):
        if min_x > frame[3 * i]:
            min_x = frame[3 * i]
        elif max_x < frame[3 * i]:
            max_x = frame[3 * i]

        if min_y > frame[3 * i + 1]:
            min_y = frame[3 * i + 1]
        elif max_y < frame[3 * i + 1]:
            max_y = frame[3 * i + 1]

        if min_z > frame[3 * i + 2]:
            min_z = frame[3 * i + 2]
        elif max_z < frame[3 * i + 2]:
            max_z = frame[3 * i + 2]
    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
    volume = volume / 1000
    return volume


def compute_feature_0(frames):
    array = []
    for frame in frames:
        array.append(compute_feature0_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by shoulders
def compute_feature_1_per_frame(frame):
    jid = 4
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 7
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(r_shoulder - neck, l_shoulder - neck)
    return angle


def compute_feature_1(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_1_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at right shoulder by neck and left shoulder
def compute_feature_2_per_frame(frame):
    jid = 4
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 7
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(neck - r_shoulder, l_shoulder - r_shoulder)
    return angle


def compute_feature_2(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_2_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at left shoulder by neck and right shoulder
def compute_feature_3_per_frame(frame):
    jid = 4
    r_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 7
    l_shoulder = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(neck - l_shoulder, r_shoulder - l_shoulder)
    return angle


def compute_feature_3(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_3_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by vertical and back
def compute_feature_4_per_frame(frame):
    jid = 3
    head = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    up = np.asarray([0.0, 1.0, 0.0])
    angle = angle_between(head - root, up)
    return angle


def compute_feature_4(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_4_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Angle at neck by head and back
def compute_feature_5_per_frame(frame):
    jid = 3
    head = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 1
    spine = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    angle = angle_between(head - neck, spine - neck)
    return angle


def compute_feature_5(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_5_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between right hand and root
def compute_feature_6_per_frame(frame):
    jid = 6
    hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(hand, root)
    return distance / 10


def compute_feature_6(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_6_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between left hand and root
def compute_feature_7_per_frame(frame):
    jid = 9
    hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(hand, root)
    return distance / 10


def compute_feature_7(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_7_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between right foot and root
def compute_feature_8_per_frame(frame):
    jid = 12
    foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(foot, root)
    return distance / 10


def compute_feature_8(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_8_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Distance between left foot and root
def compute_feature_9_per_frame(frame):
    jid = 15
    foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    distance = distance_between(foot, root)
    return distance / 10


def compute_feature_9(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_9_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Area of triangle between hands and neck
def compute_feature_10_per_frame(frame):
    jid = 9
    l_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 2
    neck = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 6
    r_hand = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    area = area_triangle(l_hand, neck, r_hand)
    return area / 100


def compute_feature_10(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_10_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Area of triangle between feet and root
def compute_feature_11_per_frame(frame):
    jid = 15
    l_foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 0
    root = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    jid = 12
    r_foot = np.asarray([frame[3 * jid], frame[3 * jid + 1], frame[3 * jid + 2]])
    area = area_triangle(l_foot, root, r_foot)
    return area / 100


def compute_feature_11(frames):
    array = []
    for frame in frames:
        array.append(compute_feature_11_per_frame(frame))
    array = np.asarray(array)
    return np.mean(array)


# Calculate speed
def calculate_speed(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    for i in range(1, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        distance = distance_between(new_position, old_position) / 10
        array.append(distance / time_step)
        old_position = new_position.copy()
    array = np.asarray(array)
    return np.mean(array)


# Speed of right hand
def compute_feature_12(frames, time_step):
    return calculate_speed(frames, time_step, 6)


# Speed of left hand
def compute_feature_13(frames, time_step):
    return calculate_speed(frames, time_step, 9)


# Speed of head
def compute_feature_14(frames, time_step):
    return calculate_speed(frames, time_step, 3)


# Speed of right foot
def compute_feature_15(frames, time_step):
    return calculate_speed(frames, time_step, 12)


# Speed of left foot
def compute_feature_16(frames, time_step):
    return calculate_speed(frames, time_step, 15)


# Calculate acceleration
def calculate_acceleration(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    new_position = np.asarray([frames[1][3 * jid], frames[1][3 * jid + 1], frames[1][3 * jid + 2]])
    old_velocity = (new_position - old_position) / time_step
    old_position = new_position.copy()
    for i in range(2, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        new_velocity = (new_position - old_position) / time_step
        acceleration = (new_velocity - old_velocity) / time_step
        acceleration_mag = np.linalg.norm(acceleration) / 10
        old_position = new_position.copy()
        old_velocity = new_velocity.copy()
        array.append(acceleration_mag)
    array = np.asarray(array)
    return np.mean(array)


# Acceleration of right hand
def compute_feature_17(frames, time_step):
    return calculate_acceleration(frames, time_step, 6)


# Acceleration of left hand
def compute_feature_18(frames, time_step):
    return calculate_acceleration(frames, time_step, 9)


# Acceleration of head
def compute_feature_19(frames, time_step):
    return calculate_acceleration(frames, time_step, 3)


# Acceleration of right foot
def compute_feature_20(frames, time_step):
    return calculate_acceleration(frames, time_step, 12)


# Acceleration of left foot
def compute_feature_21(frames, time_step):
    return calculate_acceleration(frames, time_step, 15)


# Calculate movement jerk
def calculate_movement_jerk(frames, time_step, jid):
    array = []
    old_position = np.asarray([frames[0][3 * jid], frames[0][3 * jid + 1], frames[0][3 * jid + 2]])
    new_position = np.asarray([frames[1][3 * jid], frames[1][3 * jid + 1], frames[1][3 * jid + 2]])
    old_velocity = (new_position - old_position) / time_step
    old_position = new_position.copy()
    new_position = np.asarray([frames[2][3 * jid], frames[2][3 * jid + 1], frames[2][3 * jid + 2]])
    new_velocity = (new_position - old_position) / time_step
    old_acceleration = (new_velocity - old_velocity) / time_step
    old_velocity = new_velocity.copy()
    old_position = new_position.copy()
    for i in range(3, len(frames)):
        new_position = np.asarray([frames[i][3 * jid], frames[i][3 * jid + 1], frames[i][3 * jid + 2]])
        new_velocity = (new_position - old_position) / time_step
        new_acceleration = (new_velocity - old_velocity) / time_step
        jerk = (new_acceleration - old_acceleration) / time_step
        jerk_mag = np.linalg.norm(jerk) / 10
        old_position = new_position.copy()
        old_velocity = new_velocity.copy()
        old_acceleration = new_acceleration.copy()
        array.append(jerk_mag)
    array = np.asarray(array)
    return np.mean(array)


# Movement jerk of right hand
def compute_feature_22(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 6)


# Movement jerk of left hand
def compute_feature_23(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 9)


# Movement jerk of head
def compute_feature_24(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 3)


# Movement jerk of right foot
def compute_feature_25(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 12)


# Movement jerk of left foot
def compute_feature_26(frames, time_step):
    return calculate_movement_jerk(frames, time_step, 15)


# Foot strike points
def calculate_foot_strike_points(frames, jid):
    strike_points = []
    for i in range(1, len(frames) - 1):
        foot_position_prev = frames[i - 1][3 * jid + 1]
        foot_position_curr = frames[i][3 * jid + 1]
        foot_position_next = frames[i + 1][3 * jid + 1]
        if foot_position_prev == foot_position_curr == foot_position_next:
            if not(i - 1 in strike_points or i in strike_points):
                strike_points.append(i)
        elif foot_position_prev >= foot_position_curr <= foot_position_next:  # lowest point of foot in trajectory
            strike_points.append(i)
    return np.asarray(strike_points)


def calculate_stride_length(frames, time_step):
    strike_points_right = calculate_foot_strike_points(frames, 12)
    strike_points_left = calculate_foot_strike_points(frames, 15)
    if len(strike_points_right) < 2 and len(strike_points_left) < 2:
        return len(frames), len(frames) * time_step
    if len(strike_points_right) < 2:
        mean_stride_length = np.mean(strike_points_left[1:] - strike_points_left[:-1] + 1)
        return mean_stride_length, mean_stride_length * time_step
    if len(strike_points_left) < 2:
        mean_stride_length = np.mean(strike_points_right[1:] - strike_points_right[:-1] + 1)
        return mean_stride_length, mean_stride_length * time_step
    mean_stride_length = np.mean(np.concatenate((strike_points_right[1:] - strike_points_right[:-1],
                                                 strike_points_left[1:] - strike_points_left[:-1]), axis=0))
    return mean_stride_length, mean_stride_length * time_step


# Stride length and gait cycle time
def compute_feature_27_28(frames, time_step):
    return calculate_stride_length(frames, time_step)


def compute_features(frames, time_step, add_stride_features=False):
    features = [
        # Volume
        compute_feature_0(frames),
        # Angles
        compute_feature_1(frames),
        compute_feature_2(frames),
        compute_feature_3(frames),
        compute_feature_4(frames),
        compute_feature_5(frames),
        # Distances
        compute_feature_6(frames),
        compute_feature_7(frames),
        compute_feature_8(frames),
        compute_feature_9(frames),
        # Areas
        compute_feature_10(frames),
        compute_feature_11(frames),
        # Speeds
        compute_feature_12(frames, time_step),
        compute_feature_13(frames, time_step),
        compute_feature_14(frames, time_step),
        compute_feature_15(frames, time_step),
        compute_feature_16(frames, time_step),
        # Accelerations
        compute_feature_17(frames, time_step),
        compute_feature_18(frames, time_step),
        compute_feature_19(frames, time_step),
        compute_feature_20(frames, time_step),
        compute_feature_21(frames, time_step),
        # Movement Jerk
        compute_feature_22(frames, time_step),
        compute_feature_23(frames, time_step),
        compute_feature_24(frames, time_step),
        compute_feature_25(frames, time_step),
        compute_feature_26(frames, time_step)
    ]
    if add_stride_features:
        stride_features = compute_feature_27_28(frames, time_step)
        features.extend(stride_features)
    return features
