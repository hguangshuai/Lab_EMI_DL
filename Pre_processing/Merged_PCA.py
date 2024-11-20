import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_and_merge_sensor_data(file_path):
    """
    Load and merge sensor data from an H5 file.

    Parameters:
        file_path (str): Path to the H5 file.

    Returns:
        tuple: A tuple containing:
            - sensor_data (ndarray): Combined sensor data from all groups.
            - labels (ndarray): Corresponding labels for the sensor data.
    """
    sensor_data = []
    labels = []

    # Open the H5 file
    with h5py.File(file_path, 'r') as h5file:
        # Iterate through all groups
        for group_name in h5file.keys():
            group = h5file[group_name]
            # Check if necessary datasets exist
            if 'con_sensor_I' in group and 'con_sensor_R' in group and 'Label' in group:
                # Combine imaginary and real parts
                con_sensor_RB = group['con_sensor_RB'][:]
                con_sensor_R = group['con_sensor_R'][:]
                combined_data = np.hstack([con_sensor_RB, con_sensor_R])
                sensor_data.append(combined_data)
                labels.extend(group['Label'][:])

    # Convert to NumPy arrays
    sensor_data = np.vstack(sensor_data)
    labels = np.array(labels)

    return sensor_data, labels

def perform_pca(sensor_data, n_components=2):
    """
    Perform PCA on the sensor data.

    Parameters:
        sensor_data (ndarray): The input sensor data.
        n_components (int): Number of PCA components to compute.

    Returns:
        ndarray: PCA-transformed data.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(sensor_data)

def plot_pca_results(pca_data, labels, title="PCA Distribution of Sensor Data"):
    """
    Plot the PCA results with labels as color coding.

    Parameters:
        pca_data (ndarray): PCA-transformed data (2D).
        labels (ndarray): Labels for color coding.
        title (str): Title of the plot.
    """
    # Normalize labels for color mapping
    norm_labels = (labels - labels.min()) / (labels.max() - labels.min())

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=norm_labels,
        cmap='coolwarm',
        edgecolor='k',
        s=50
    )
    plt.colorbar(scatter, label='Compressive Strength (Normalized)')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main workflow
if __name__ == "__main__":
    # Path to the H5 file
    file_path = "Data/Merged/Surface_all.h5"

    # Step 1: Load and merge sensor data
    sensor_data, labels = load_and_merge_sensor_data(file_path)

    # Step 2: Perform PCA
    pca_results = perform_pca(sensor_data, n_components=2)

    # Step 3: Plot PCA results
    plot_pca_results(pca_results, labels)
