import numpy as np


def to_polar(points):
    radius = np.sqrt(points[..., 0]**2 + points[..., 1]**2)
    angle = np.arctan2(points[..., 1], points[..., 0])
    
    return radius, np.mod(angle, 2 * np.pi)


def to_cartesian(radius, angle):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    
    return np.stack([x, y], axis=-1)


def rotation(angle):
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    
    return rot


def make_line(start_point, stop_point):
    center = start_point
    normal = stop_point - start_point
    length = np.linalg.norm(normal, axis=-1)
    normal = normal / np.expand_dims(length, axis=-1)
    
    return center, normal, length


def point_line_distance(center, normal, points):
    # looking along the direction of the line, the right side has positive, the left negative
    # distance
    
    diff = points - center
    distance = diff[..., 0] * normal[1] - diff[..., 1] * normal[0]
    
    return distance


def point_line_distance_along_line(center, normal, points):
    # the signed distance along the line from the center in the direction of the normal where
    # the points are closest to it
    
    distance = np.sum((points - center) * normal, axis=-1)
    
    return distance


def orthogonal_vector(vector):
    vector = rotation(np.pi / 2) @ vector
    
    return vector


def circle_line_intersection(circle_center, circle_radius, line_center, line_normal):
    # the interection point where the line passes the circles center orthogonally
    intersect = line_line_intersection(line_center, line_normal, 
                                       circle_center, orthogonal_vector(line_normal))
    
    return intersect


def line_line_intersection(center, normal, other_centers, other_normals):
    denominator = normal[0] * other_normals[..., 1] - normal[1] * other_normals[..., 0]
    diff = other_centers - center
    
    t = (diff[..., 0] * other_normals[..., 1] - diff[..., 1] * other_normals[..., 0]) / denominator
    # other_t = (diff[..., 0] * normal[1] - diff[..., 1] * normal[0]) / denominator
    
    intersection = center + t * normal
    
    return intersection


def angle(normal, other_normals):    
    return np.arccos(np.clip(np.sum(normal * other_normals, axis=-1), -1, 1))


def is_parallel(normal_1, normal_2):
    a = angle(normal_1, normal_2)
    eps = 1e-7

    if np.abs(a) > eps and np.abs(a) < np.pi - eps:
        return False
    else:
        return True
    
    
def point_within_bounded_line(center, normal, lower_bound, upper_bound, points):
    t = point_line_distance_along_line(center, normal, points)
    
    arg = (lower_bound <= t) & (t <= upper_bound)
    
    return arg
    
    
def point_bounded_line_distance(center, normal, lower_bound, upper_bound, points):
    upper_point = center + upper_bound * normal
    lower_point = center + lower_bound * normal
    t = point_line_distance_along_line(center, normal, points)
    
    arg_lower = t < lower_bound
    arg_upper = upper_bound < t
    arg = ~arg_lower & ~arg_upper
    
    distance = np.zeros(points.shape[:-1])
    
    distance[arg] = np.abs(point_line_distance(center, normal, points[arg]))
    distance[arg_lower] = np.linalg.norm(points[arg_lower] - lower_point, axis=-1)
    distance[arg_upper] = np.linalg.norm(points[arg_upper] - upper_point, axis=-1)
    
    return distance
    
    
def angle_along_bounded_line(center, normal, lower_bound, upper_bound, points):
    # the angle along a bounded line where 0 is along the normal
    
    distance = point_line_distance_along_line(center, normal, points)
    sign = np.sign(point_line_distance(center, normal, points))
    sign = np.where(sign == 0, 1, sign)
    angles = np.zeros(distance.shape)
    
    arg_lower = distance < lower_bound
    arg_upper = upper_bound < distance
    arg = ~arg_lower & ~arg_upper
        
    angles[arg] = sign[arg] * -np.pi / 2
        
    upper_point = center + upper_bound * normal
    upper_vectors = points[arg_upper] - upper_point
    upper_vectors = upper_vectors / np.linalg.norm(upper_vectors, axis=-1, keepdims=True)
    
    angles[arg_upper] = angle(normal, upper_vectors) * -sign[arg_upper]
        
    lower_point = center + lower_bound * normal
    lower_vectors = points[arg_lower] - lower_point
    lower_vectors = lower_vectors / np.linalg.norm(lower_vectors, axis=-1, keepdims=True)
    
    angles[arg_lower] = angle(normal, lower_vectors) * -sign[arg_lower]
        
    angles = np.mod(angles, 2 * np.pi)
        
    return angles


def circumfence_along_bounded_line(center, normal, lower_bound, upper_bound, width, points):
    angles = angle_along_bounded_line(center, normal, lower_bound, upper_bound, points)
    t = point_line_distance_along_line(center, normal, points)
    distance = np.zeros(angles.shape)
                
    distance += angles * width
    
    arg_smaller = angles < np.pi / 2
    arg_bigger = angles > np.pi / 2
    arg_exact = ~arg_smaller & ~arg_bigger
    
    distance[arg_exact] += upper_bound - t[arg_exact]
    distance[arg_bigger] += upper_bound - lower_bound
    
    arg_smaller = angles < np.pi * 3 / 2
    arg_bigger = angles > np.pi * 3 / 2
    arg_exact = ~arg_smaller & ~arg_bigger
    
    distance[arg_exact] += t[arg_exact] - lower_bound
    distance[arg_bigger] += upper_bound - lower_bound
                        
    return distance