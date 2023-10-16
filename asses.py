import pandas as pd
import numpy as np
import json
from pathlib import Path

from horse_racing.track.track import RaceTrack


class Context:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        
        if self.data_path.is_file():
            with open(self.data_path) as f:
                data = json.load(f)
                
        else:
            data = dict()
            
        self.data = data
        
    def has_entry(self, key):
        return key in self.data
        
    def __call__(self, key, value):
        self.data[key] = value
        
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exec_value, exec_traceback):
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f)
            
        return True


track_dir_path = "./track_data/AQU"
tracking_csv_path = "./csv/nyra_tracking_table.csv"
race_csv_path = "./csv/nyra_race_table.csv"
output_json_path = "./csv/eval.json"
track_id = "AQU"
course_type = "D"
from_start = False
    
tracking_data = pd.read_csv(tracking_csv_path)
race_data = pd.read_csv(race_csv_path)
race_track = RaceTrack.from_directory(track_dir_path)

data = tracking_data[tracking_data["track_id"] == track_id]

with Context(output_json_path) as data_context:
    for race_date in data["race_date"].unique():
        race_date_data = data[data["race_date"] == race_date]
        
        for race_number in race_date_data["race_number"].unique():
            race_number_data = race_date_data[race_date_data["race_number"] == race_number]
            
            this_race_data = race_data[(race_data["track_id"] == track_id) &
                                       (race_data["race_date"] == race_date) &
                                       (race_data["race_number"] == race_number)]
            if this_race_data["course_type"].item() != course_type:
                continue
            
            key = f"{track_id}-{race_date}-{race_number}-{course_type}"
            if data_context.has_entry(key) and not from_start:
                continue
            
            trajectories = []
            
            for program_number in race_number_data["program_number"].unique():
                program_number_data = race_number_data[race_number_data["program_number"] == program_number]
                
                lat = program_number_data['latitude'].to_numpy()
                long = program_number_data['longitude'].to_numpy()
                index = program_number_data['trakus_index'].to_numpy()
                
                arg = np.argsort(index)
                lat = lat[arg]
                long = long[arg]
                
                coords = np.stack([lat, long], axis=-1)
                trajectories.append(coords)
                
            trajectories = np.array(trajectories)
            
            race_track.plot_race(trajectories, course_type)
            
            value = None
            while not value:
                x = input(f"Use Track {track_id}, {race_date}, {race_number}, {course_type}? ")
                
                if x == "n":
                    value = {"use": False, "run_in": None}
                elif x == "y":
                    value = {"use": True, "run_in": None}
                elif x == "r":
                    value = {"use": True, "run_in": course_type}
                elif x == "q":
                    raise KeyboardInterrupt()
                else:
                    value = True
            
            if type(value) is dict:
                data_context(key, value)