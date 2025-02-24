import numpy as np
import csv


class SimulationLogger:
    def __init__(self):
        # Dictionary to store different types of data (positions, velocities, joint angles, etc.)
        self.data = {}

    def log(self, key, value):
        """
        Log data for a given key. Each key corresponds to a particular type of data (e.g., 'position', 'velocity').
        Value should be an array-like object that is logged at each timestep.
        """
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def get_logged_data(self, key):
        """
        Retrieve the logged data for a specific key.
        """
        return np.array(self.data.get(key, []))

    def save_to_csv(self, filename, sim_time):
        """
        Save the logged data to a CSV file.
        :param filename: Name of the CSV file to save
        :param sim_time: Simulation time array to include in the CSV file
        """
        # Combine simulation time with the logged data
        data_to_save = {'Time': sim_time}

        # Flatten the logged data into columns
        for key, values in self.data.items():
            values = np.array(values)
            if values.ndim > 1:  # If the data is multi-dimensional (e.g., position x, y, z)
                for i in range(values.shape[1]):
                    data_to_save[f'{key}_{i}'] = values[:, i]
            else:
                data_to_save[key] = values

        # Write to CSV
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data_to_save.keys())
            writer.writeheader()
            for i in range(len(sim_time)):
                row = {key: data_to_save[key][i] for key in data_to_save}
                writer.writerow(row)

    def clear(self):
        """
        Clear all logged data.
        """
        self.data.clear()