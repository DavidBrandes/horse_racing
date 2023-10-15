import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from track import RaceTrack


track_dir_path = "./track_data/AQU"
tracking_csv_path = "./csv/nyra_tracking_table.csv"

tracking_data = pd.read_csv(tracking_csv_path)

track_id = "AQU"
course_type = "D"
race_date = "2019-01-01"
race_number = 8
race_length = 1810.52

data = tracking_data
data = data[data["track_id"] == track_id]
# data = data[data["course_type"] == course_type]
data = data[data["race_date"] == race_date]
data = data[data["race_number"] == race_number]

trajectories = []
for program_number in data["program_number"].unique():
    program_data = data[data["program_number"] == program_number]
    
    lat = program_data['latitude'].to_numpy()
    long = program_data['longitude'].to_numpy()
    index = program_data['trakus_index'].to_numpy()
    
    arg = np.argsort(index)
    lat = lat[arg]
    long = long[arg]
    
    coords = np.stack([lat, long], axis=-1)
    trajectories.append(coords)
    
    
trajectories = np.array(trajectories)
# trajectories = trajectories[:, ::-1]
trajectories = trajectories[0]
race_track = RaceTrack.from_directory(track_dir_path)

race_track.plot_race(trajectories, "main", "D", "D", colorize_by="distance")
