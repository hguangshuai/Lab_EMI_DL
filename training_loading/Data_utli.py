import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class EMIDataLoader:
    """
    A data loader class to handle and merge EMI data from an H5 file.
    """

    def __init__(self, file_name):
        """
        Initialize the data loader with the H5 file.

        Parameters:
            file_name (str): Path to the H5 file.
        """
        self.file_name = file_name
        self.data = {}
        self.labels = []

    def load_data(self):
        """
        Load and merge all data from the H5 file.

        Attributes:
            data (dict): A dictionary where each key is a dataset name (e.g., 'con_sensor_I')
                         and the value is the merged data list.
            labels (ndarray): The combined labels from all groups.
        """
        with h5py.File(self.file_name, 'r') as h5file:
            for group_name in h5file.keys():
                group = h5file[group_name]
                for dataset_name in group.keys():
                    dataset_values = group[dataset_name][:]
                    
                    # Merge data by dataset name (store in a list for varying sizes)
                    if dataset_name not in self.data:
                        self.data[dataset_name] = []
                    self.data[dataset_name].append(dataset_values)
                
                # Merge labels if available
                if 'Label' in group:
                    self.labels.extend(group['Label'][:])

        # Convert lists of arrays to single arrays (concatenate along the first axis)
        for dataset_name in self.data:
            self.data[dataset_name] = np.concatenate(self.data[dataset_name], axis=0)

        # Convert labels to NumPy array
        self.labels = np.array(self.labels)

    def get_features_and_labels(self):
        """
        Prepare features (X) and labels (y) for training/testing.

        Returns:
            tuple: A tuple containing:
                - X (ndarray): The combined feature array.
                - y (ndarray): The corresponding labels array.
        """
        if 'con_sensor_I' not in self.data or 'con_sensor_R' not in self.data:
            raise ValueError("The required datasets 'con_sensor_I' and 'con_sensor_R' are not present in the file.")

        # Combine imaginary and real parts as features
        X = np.hstack([self.data['con_sensor_I'], self.data['con_sensor_R']])

        # Ensure labels are available
        if len(self.labels) == 0:
            raise ValueError("Labels are not available in the dataset.")

        y = self.labels
        return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
        X (ndarray): The feature array.
        y (ndarray): The label array.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Training and testing data as (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Main workflow
if __name__ == "__main__":
    # Path to the H5 file
    file_name = "Data/Merged/Surface_all.h5"

    # Initialize and load data
    loader = EMIDataLoader(file_name)
    loader.load_data()

    # Get features and labels
    X, y = loader.get_features_and_labels()

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Display dataset shapes
    print("Training Features Shape:", X_train.shape)
    print("Training Labels Shape:", y_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Testing Labels Shape:", y_test.shape)
