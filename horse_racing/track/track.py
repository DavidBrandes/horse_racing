import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import json

from horse_racing.utils import fitting
from horse_racing.utils import geometry
from horse_racing.track import shapes
from horse_racing.utils import geography


class RaceTrack:
    def __init__(self, origin, side_coords_1, side_coords_2, circle_coords_1, circle_coords_2,
                 outline_coords_dict, finish_line_coords_dict, run_ins_dict={}):
        self._origin = origin
        
        side_points_1 = geography.to_xy(*side_coords_1.T, *origin)
        side_points_2 = geography.to_xy(*side_coords_2.T, *origin)
        circle_points_1 = geography.to_xy(*circle_coords_1.T, *origin)
        circle_points_2 = geography.to_xy(*circle_coords_2.T, *origin)
        outline_points_dict = {
            key: geography.to_xy(*item.T, *origin) for key, item in outline_coords_dict.items()
        }
        finish_line_points_dict = {
            key: geography.to_xy(*item.T, *origin) for key, item in finish_line_coords_dict.items()
        }
        
        # the lines corresponding to the two sides
        (center_1, normal_1), (center_2, normal_2) = fitting.fit_parallel_lines([side_points_1, 
                                                                                 side_points_2])
        
        # the central line
        center = (center_1 + center_2) / 2
        normal = normal_1
        
        # the one circle
        origin_1, radius_1 = fitting.fit_circle_along_lines(center_1, normal_1, 
                                                            center_2, normal_2, 
                                                            circle_points_1)
        center_offset_1 = geometry.point_line_distance_along_line(center, normal, origin_1)
        
        # the other circle
        origin_2, radius_2 = fitting.fit_circle_along_lines(center_1, normal_1, 
                                                            center_2, normal_2, 
                                                            circle_points_2)
        center_offset_2 = geometry.point_line_distance_along_line(center, normal, origin_2)
        
        # orient the circles along the normals
        if center_offset_1 < center_offset_2:
            lower_bound = center_offset_1
            upper_bound = center_offset_2
            
        else:
            lower_bound = center_offset_2
            upper_bound = center_offset_1
            
        self._normal = normal
        self._center = center
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
            
        self._circuit = shapes.Circuit(center, normal, lower_bound, upper_bound)
        
        # create the track widths and radius
        track_widths, track_radius = {}, {}
        
        for track_id, points in outline_points_dict.items():
            distance = geometry.point_bounded_line_distance(center, normal, 
                                                            lower_bound, upper_bound, 
                                                            points)
            distance = np.sort(distance)
            
            track_widths[track_id] = np.diff(distance)[0]
            track_radius[track_id] = distance[0]
                
        self._track_radius = track_radius
        self._track_widths = track_widths
            
        self._finish_line_points = finish_line_points_dict
        
        # create the run in shapes
        run_ins = {}
        
        for run_in_id, run_in in run_ins_dict.items():
            start_point = geography.to_xy(*run_in["start_coords"].T, *origin)
            end_point = geography.to_xy(*run_in["end_coords"].T, *origin)
            
            if run_in["along_side"]:
                d = geometry.point_line_distance(center, normal, end_point)
                d = -np.sign(d) * track_radius[run_in["track_id"]]
                run_in_center = center + geometry.orthogonal_vector(normal) * d
                run_in_normal = normal
                                    
            else:
                points = geography.to_xy(*run_in["coords"].T, *origin)
                points = np.vstack([points, start_point, end_point])
                run_in_center, run_in_normal = fitting.fit_line(points)
                                
                d = track_radius[run_in["track_id"]]
                intersect_1 = geometry.circle_line_intersection(origin_1, d,
                                                                run_in_center, run_in_normal)
                intersect_2 = geometry.circle_line_intersection(origin_2, d,
                                                                run_in_center, run_in_normal)
                
                side_1_center = center + geometry.orthogonal_vector(normal) * d
                side_2_center = center - geometry.orthogonal_vector(normal) * d
                
                intersect_3 = geometry.line_line_intersection(side_1_center, normal,
                                                              run_in_center, run_in_normal)
                intersect_4 = geometry.line_line_intersection(side_2_center, normal,
                                                              run_in_center, run_in_normal)
                
                point, min_dist = None, np.inf
                
                for intersect in [intersect_1, intersect_2, intersect_3, intersect_4]:
                    dist = np.linalg.norm(end_point - intersect)
                    if dist < min_dist:
                        point = intersect
                        min_dist = dist
                        
                end_point = point
                
            t_end = geometry.point_line_distance_along_line(run_in_center, run_in_normal, 
                                                            end_point)
            t_start = geometry.point_line_distance_along_line(run_in_center, run_in_normal, 
                                                              start_point)
        
            if t_start > t_end:
                run_in_normal = -run_in_normal
                t_start, t_end = -t_start, -t_end
                
            start_point = run_in_center + t_start * run_in_normal
            end_point = run_in_center + t_end * run_in_normal

            run_ins[run_in_id] = shapes.StraightRunIn(run_in_normal, start_point, end_point)
            
        self._run_ins = run_ins
    
    @classmethod
    def from_directory(cls, dir_path):
        dir_path = Path(dir_path)
        
        side_coords_1 = np.loadtxt(dir_path / "side_1.txt", delimiter=",")
        side_coords_2 = np.loadtxt(dir_path / "side_2.txt", delimiter=",")
        circle_coords_1 = np.loadtxt(dir_path / "circle_1.txt", delimiter=",")
        circle_coords_2 = np.loadtxt(dir_path / "circle_2.txt", delimiter=",")
        origin = np.loadtxt(dir_path / "origin.txt", delimiter=",")
        
        outline_coords_dict, end_coords_dict, run_ins_dict = {}, {}, {}
        for file in dir_path.glob("*.txt"):
            name_parts = file.name.split(".")[0].split("_")
            
            if len(name_parts) != 2:
                continue
            
            if name_parts[0] == "outline":
                outline_coords_dict[name_parts[1]] = np.loadtxt(file, delimiter=",")
                
            elif name_parts[0] == "finish":
                end_coords_dict[name_parts[1]] = np.loadtxt(file, delimiter=",")
                
            elif name_parts[0] == "runin":
                json_path = dir_path / ("_".join(name_parts) + ".json")
                with open(json_path) as f:
                    data = json.load(f)
                    
                key = name_parts[1]
                value = {"start_coords": np.array(data["start_coords"]), 
                         "end_coords": np.array(data["end_coords"]), 
                         "along_side": data["along_side"],
                         "track_id": data["track_id"],
                  }
                
                if not value["along_side"]:
                    value["coords"] = np.loadtxt(file, delimiter=",")
                    
                run_ins_dict[key] = value
        
        return cls(origin, side_coords_1, side_coords_2, circle_coords_1, circle_coords_2,
                   outline_coords_dict, end_coords_dict, run_ins_dict)
    
    def _plot_finish_line(self, ax, finish_line_id, track_id):
        radius, width = self._track_radius[track_id], self._track_widths[track_id]
        point = self._finish_line_points[finish_line_id]
        
        angle = geometry.angle_along_bounded_line(self._center, self._normal, 
                                                  self._lower_bound, self._upper_bound, 
                                                  point)
        t = geometry.point_line_distance_along_line(self._center, self._normal, point)
        t = np.clip(t, self._lower_bound, self._upper_bound)
        
        center = self._center + t * self._normal
        normal = (geometry.rotation(angle) @ self._normal.reshape(2, 1)).reshape(2)
        
        inner_point = center + radius * normal
        outer_point = center + (radius + width) * normal
        
        ax.plot(*zip(inner_point, outer_point), c="k", linestyle="dotted")
        
    def _evaluate(self, points, track_id, finish_line_id="main", run_in_id=None):        
        return self._circuit(points, self._finish_line_points[finish_line_id], 
                             self._track_radius[track_id],
                             self._run_ins[run_in_id] if run_in_id else None)
        
    def plot_track(self):
        fig, ax = plt.subplots()
        
        for track_id in self._track_radius.keys():                
            self._circuit.plot(ax, self._track_radius[track_id], self._track_widths[track_id])
            
            for finish_line_id in self._finish_line_points.keys():
                self._plot_finish_line(ax, finish_line_id, track_id)
                
        for run_in in self._run_ins.values():
            run_in.plot(ax)
                        
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        ax.set_title("Race Track")
        
        plt.show()
    
    def plot_race(self, coords, track_id, finish_line_id="main", run_in_id=None, colorize_by=None):
        fig, ax = plt.subplots()
                                
        self._circuit.plot(ax, self._track_radius[track_id], self._track_widths[track_id])
            
        if run_in_id:  
            self._run_ins[run_in_id].plot(ax)
        
        self._plot_finish_line(ax, finish_line_id, track_id)

        min_value, max_value = np.inf, -np.inf
        
        points = geography.to_xy(coords[..., 0], coords[..., 1], *self._origin)
        if points.ndim == 2:
            points = points.reshape(1, -1, 2)
        distance, offset, curvature = self._evaluate(points, track_id, finish_line_id, run_in_id)
                    
        if colorize_by == "distance":
            colors = distance[..., :-1]
        elif colorize_by == "offset":
            colors = offset[..., :-1]
        elif colorize_by == "curvature":
            colors = curvature[..., :-1]
        else:
            colors = np.broadcast_to(np.arange(len(points)).reshape(-1, 1), distance.shape)
        
        min_value, max_value = np.min(colors), np.max(colors)
        points = np.expand_dims(points, axis=2)
        segments = np.concatenate([points[:, :-1], points[:, 1:]], axis=2).reshape(-1, 2, 2)
        colors = colors.reshape(-1)
                
        lc = LineCollection(segments, array=colors, cmap="viridis_r", 
                            norm=plt.Normalize(min_value, max_value))
        ax.add_collection(lc)
        
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        if colorize_by in ['distance', "offset", "curvature"]:
            label = colorize_by.capitalize()
            ax.set_title(f"Race Track with {label}")
            fig.colorbar(lc, label=label)
        else:
            ax.set_title("Race Track")
        
        # TODO reset
        plt.show(block=True)
        
    def circumfence(self, track_id):
        return self._circuit.circumfence(self._track_radius[track_id])
        
    def __call__(self, coords, track_id, finish_line_id="main", run_in_id=None):
        points = geography.to_xy(coords[..., 0], coords[..., 1], *self._origin)
        
        return self._evaluate(points, track_id, finish_line_id, run_in_id)
        
        
        
        