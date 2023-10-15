import numpy as np
from scipy.optimize import least_squares

import geometry


def fit_vector(points):
    def f(a):
        return points[:, 0] * np.sin(a) - points[:, 1] * np.cos(a)
    
    def d_f(a):
        return (points[:, 0] * np.cos(a) + points[:, 1] * np.sin(a)).reshape(-1, 1)
    
    res = least_squares(f, np.zeros(1), d_f, method="lm")
    
    normal = np.stack([np.cos(res.x[0]), np.sin(res.x[0])], axis=-1)
        
    return normal


def fit_line(points):
    center = np.mean(points, axis=0)
    points = points - center
    normal = fit_vector(points)
            
    return center, normal


def fit_circle(points, x=None, y=None, radius=None):
    v0 = np.mean(points, axis=0)

    arg = []
    if x is None:
        arg.append(0)
    else:
        v0[0] = x
    if y is None:
        arg.append(1)
    else:
        v0[1] = y

    def unpack(v):
        v_ = v0.copy()
        v_[arg] = v
                
        return v_
        
    def f(v):
        v = unpack(v)
        d = np.linalg.norm(v - points, axis=-1)
        
        if radius is None:        
            return d - np.mean(d)
        else:
            return d - radius
    
    def d_f(v):
        v = unpack(v)
        d = np.linalg.norm(v - points, axis=-1)
        
        d_d = np.zeros(points.shape)
        arg_nonzero = d != 0
        d_d[arg_nonzero, 0] = (v[0] - points[arg_nonzero, 0]) / d[arg_nonzero]
        d_d[arg_nonzero, 1] = (v[1] - points[arg_nonzero, 1]) / d[arg_nonzero]
        d_d = d_d[:, arg]
        
        if radius is None:
            return d_d - np.mean(d_d, axis=0)
        else:
            return d_d
    
    res = least_squares(f, v0[arg], d_f, method='lm')
    v = unpack(res.x)
    
    center = v
    if radius is None:
        radius = np.mean(np.linalg.norm(v - points, axis=-1))
    
    return center, radius


def fit_orthogonal_line(center, normal, points):
    other_center = np.mean(points, axis=0)
    other_normal = geometry.orthogonal_vector(normal)
    
    return other_center, other_normal


def fit_parallel_lines(points_list):
    center_list = [np.mean(points, axis=0) for points in points_list]
    points_list = [points - center for points, center in zip(points_list, center_list)]
    
    normal = fit_vector(np.concatenate(points_list, axis=0))    
    normal_list = [normal for points in points_list]
        
    return list(zip(center_list, normal_list))


def fit_circle_along_lines(center_1, normal_1, center_2, normal_2, points):    
    if not geometry.is_parallel(normal_1, normal_2):
        radius = None
        
        raise NotImplementedError("Can only fit parallel lines for now")
    
    else:
        distance = geometry.point_line_distance(center_1, normal_1, center_2)
        radius = np.abs(distance) / 2
        
    cirlce_normal = normal_1
    cirlce_center = (center_1 + center_2) / 2
    
    R = np.array([[cirlce_normal[0], -cirlce_normal[1]],
                  [cirlce_normal[1], cirlce_normal[0]]])
    p = cirlce_center
    
    points = np.einsum("ab, cb -> ca", R.T, points - p)
    
    center, radius = fit_circle(points, y=0, radius=radius)
    center = R @ center + p
            
    return center, radius        
    
    
    
    
    