import h5py
import numpy as np
import matplotlib.pyplot as plt

# File path (update this if the file location changes)
file_path = 'Data/Separated/Concrete_Sur/350 Slag-1.h5'

# Load the H5 file
with h5py.File(file_path, 'r') as h5file:
    # Read datasets from the file
    age = h5file['Age'][:]  # Concrete curing age (in days)
    label = h5file['Label'][:]  # Compressive strength (MPa)
    temp = h5file['Temp'][:]  # Temperature (in °C)
    con_sensor_I = h5file['con_sensor_I'][:]  # Real-time signal, imaginary part
    con_sensor_IB = h5file['con_sensor_IB'][:]  # Baseline signal, imaginary part
    con_sensor_R = h5file['con_sensor_R'][:]  # Real-time signal, real part
    con_sensor_RB = h5file['con_sensor_RB'][:]  # Baseline signal, real part

# Define the frequency range corresponding to the 100 points
frequency = np.linspace(10, 1000, 100)  # From 10 kHz to 1000 kHz

# Print basic information
print("Age (Curing Days):", age)
print("Compressive Strength (MPa):", label)
print("Temperature (°C):", temp)

# Visualize the sensor data for the first sample
sample_index = 0  # Choose a sample to visualize

plt.figure(figsize=(12, 8))

# Real-time signal (Imaginary and Real parts)
plt.subplot(2, 2, 1)
plt.plot(frequency, con_sensor_I[sample_index], label='Imaginary Part')
plt.plot(frequency, con_sensor_R[sample_index], label='Real Part')
plt.title(f"Real-Time Signal (Sample {sample_index})")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Signal Amplitude")
plt.legend()

# Baseline signal (Imaginary and Real parts)
plt.subplot(2, 2, 2)
plt.plot(frequency, con_sensor_IB[sample_index], label='Imaginary Part')
plt.plot(frequency, con_sensor_RB[sample_index], label='Real Part')
plt.title(f"Baseline Signal (Sample {sample_index})")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Signal Amplitude")
plt.legend()

# Compressive strength vs age
plt.subplot(2, 2, 3)
plt.plot(age, label, marker='o', linestyle='-')
plt.title("Compressive Strength vs Curing Age")
plt.xlabel("Curing Age (Days)")
plt.ylabel("Compressive Strength (MPa)")

# Temperature vs age
plt.subplot(2, 2, 4)
plt.plot(age, temp, marker='o', linestyle='-')
plt.title("Temperature vs Curing Age")
plt.xlabel("Curing Age (Days)")
plt.ylabel("Temperature (°C)")

plt.tight_layout()
plt.show()
