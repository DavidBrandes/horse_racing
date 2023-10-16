import numpy as np
from matplotlib import patches

from horse_racing.utils import geometry

                
class Circuit:
    def __init__(self, center, normal, lower_bound, upper_bound):
        self._center = center
        self._normal = normal
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._length = upper_bound - lower_bound
        
    def plot(self, ax, radius, width=None, solid=True):
        orthogonal_normal = geometry.orthogonal_vector(self._normal)
                
        if width:
            widths = [radius, radius + width]
        else:
            widths = [radius]
        linestyle = "solid" if solid else "dashed"
            
        for width in widths:
            start_point = self._center + self._lower_bound * self._normal + orthogonal_normal * width
            end_point = self._center + self._upper_bound * self._normal + orthogonal_normal * width
            ax.plot(*zip(start_point, end_point), color="k", linestyle=linestyle)
    
            start_point = self._center + self._lower_bound * self._normal - orthogonal_normal * width
            end_point = self._center + self._upper_bound * self._normal - orthogonal_normal * width
            ax.plot(*zip(start_point, end_point), color="k", linestyle=linestyle)
            
            center = self._center + self._lower_bound * self._normal
            theta = geometry.angle(np.array([1, 0]), orthogonal_normal)
            theta = theta * 180 / np.pi
            arc = patches.Arc(center, 2 * width, 2 * width,
                              theta1=theta, theta2=theta + 180, 
                              color="k", linewidth=1.5, linestyle=linestyle)
            ax.add_patch(arc)
            
            center = self._center + self._upper_bound * self._normal
            theta = geometry.angle(np.array([1, 0]), orthogonal_normal)
            theta = theta * 180 / np.pi
            arc = patches.Arc(center, 2 * width, 2 * width,
                              theta1=theta - 180, theta2=theta, 
                              color="k", linewidth=1.5, linestyle=linestyle)
            ax.add_patch(arc)
        
    def circumfence(self, radius):
        return 2 * np.pi * radius + 2 * self._length
                
    def __call__(self, points, end_point, radius, run_in=None):
        point_distance = geometry.circumfence_along_bounded_line(self._center, self._normal, 
                                                                 self._lower_bound, self._upper_bound, 
                                                                 radius, points)
        end_distance = geometry.circumfence_along_bounded_line(self._center, self._normal, 
                                                               self._lower_bound, self._upper_bound, 
                                                               radius, end_point)
        
        angles = geometry.angle_along_bounded_line(self._center, self._normal, 
                                                   self._lower_bound, self._upper_bound, 
                                                   points)
        angles = np.unwrap(angles)
        clockwise = angles[..., [0]] > angles[..., [-1]]
        traversals = np.max(np.abs(angles - angles[..., [0]]) // (2 * np.pi), axis=-1, keepdims=True)
        
        circumfence = self.circumfence(radius)
        point_distance = np.where(clockwise, circumfence - point_distance, point_distance)
        end_distance = np.where(clockwise, circumfence - end_distance, end_distance)
        
        distance = end_distance - point_distance
        distance = np.mod(distance, circumfence)
        distance = distance + traversals * circumfence
        distance = np.unwrap(distance, period=circumfence)
                        
        offset = geometry.point_bounded_line_distance(self._center, self._normal, 
                                                      self._lower_bound, self._upper_bound, 
                                                      points)
        offset = offset - radius
        
        arg = geometry.point_within_bounded_line(self._center, self._normal, 
                                                 self._lower_bound, self._upper_bound, 
                                                 points)
        curvature = np.zeros(points.shape[:-1])
        curvature[~arg] = 1 / radius
        
        # without it we get the default circuit only values (in case something below is wrong)
        if run_in:
            run_in_angle = geometry.angle_along_bounded_line(self._center, self._normal, 
                                                             self._lower_bound, self._upper_bound, 
                                                             run_in.point)
            run_in_arg = np.where(clockwise, angles > run_in_angle, angles < run_in_angle)
            run_in_distance = geometry.circumfence_along_bounded_line(self._center, self._normal, 
                                                                      self._lower_bound, self._upper_bound, 
                                                                      radius, run_in.point)
            run_in_traversals = np.abs(angles[..., [-1]] - run_in_angle) // (2 * np.pi)
            
            run_in_distance = np.where(clockwise, circumfence - run_in_distance, run_in_distance)
        
            run_in_distance = end_distance - run_in_distance
            run_in_distance = np.mod(run_in_distance, circumfence)
            run_in_distance = run_in_distance + run_in_traversals * circumfence
            
            run_in_distance_, run_in_offset, run_in_curvature = run_in(points)
        
            distance[run_in_arg] = (run_in_distance + run_in_distance_)[run_in_arg]
            offset[run_in_arg] = run_in_offset[run_in_arg]
            curvature[run_in_arg] = run_in_curvature[run_in_arg]
        
        return distance, offset, curvature
    
    
class StraightRunIn:
    def __init__(self, normal, start_point, end_point):
        # the normal is directed towards the race course
        self._normal = normal
        self._start_point = start_point
        self._end_point = end_point
        self._length = np.linalg.norm(start_point - end_point)
        self.point = end_point
        
    def length(self):
        return self._length
    
    def plot(self, ax, solid=True):
        ax.plot(*zip(self._end_point, self._start_point), 
                color="grey", linestyle="solid" if solid else "dashed")
    
    def __call__(self, points):
        curvature = np.zeros(points.shape[:-1])
        offset = geometry.point_line_distance(self._end_point, self._normal, points)
        distance = geometry.point_line_distance_along_line(self._end_point, -self._normal, points)
        
        return distance, offset, curvature
    
            
            
            
            
            