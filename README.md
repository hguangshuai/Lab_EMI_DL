# Lab EMI DL: Deep Learning for Concrete Strength Sensing

This repository supports ongoing research in using deep learning and piezoelectric-based Electro-Mechanical Impedance (EMI) signals for concrete strength monitoring. The project focuses on fusing piezoelectric sensor data with machine learning models to predict concrete compressive strength in both laboratory and field environments.

## Background

Concrete compressive strength is a critical factor for the durability and safety of modern infrastructure. Traditional testing methods, while effective, are labor-intensive and time-consuming, lacking real-time monitoring capabilities. This research leverages piezoelectric sensors and machine learning to address these limitations. By combining EMI signals with advanced 1D Convolutional Neural Networks (1D CNNs), we have developed robust models for both laboratory and field applications.

For a detailed explanation of the methodology and results, refer to our published [INDOT report](https://rosap.ntl.bts.gov/view/dot/54753).

## Repository Structure

```
Lab_EMI_DL/
├── Data/             # Data folder containing merged and separated files
│   ├── Merged/       # H5 files containing combined EMI data for training/testing
│   ├── Separated/    # H5 files organized by test type and individual sensor data
├── training_loading/ # Python scripts for data loading and processing
├── Pre_processing/   # Scripts for preprocessing the data
├── models/           # Pre-trained and example models (when available)
└── README.md         # This document
```

### Data

1. **Merged**: Contains aggregated H5 files combining data from multiple sensors and experiments.
2. **Separated**: Holds H5 files divided by sensor type, test type, and other categories for targeted analysis.

### Example Data Details

- **Signal Types**:
  - **Baseline Signal**: EMI signal captured at the 4-hour curing mark.
  - **Real-Time Signal**: EMI signal captured at various curing ages, reflecting changes in the material's strength.
- **Curing Ages**: Data captured at 6 hours, 8 hours, 1 day, 7 days, and beyond.
- **Labels**: Compressive strength (in MPa) associated with each data point.

For full data categorization, refer to the `INDOT report`.

## Usage

### Data Loading
You can use the provided `EMI_loader` class to load and preprocess the data for training and evaluation.

#### Example Code
```python
from training_loading.Data_utli import EMI_loader

# Load data from a merged H5 file
loader = EMI_loader("Data/Merged/example.h5")
features = loader.fea_spec_logage()  # Generate features
labels = loader.Label  # Compressive strength values
```

### Train/Test Splitting
The repository includes utilities to split data into training and testing sets. Example:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
```

## Data and Code Availability

**Note**: This repository does not include all data or training codes. The detailed datasets and training scripts related to embedded sensors are currently restricted due to their proprietary nature and are under a **technology transfer phase**. We are committed to gradually updating this repository to aid the development of concrete sensor technology.

## Future Updates

- Integration of more detailed datasets.
- Examples of transfer learning on new sensor types.
- Expanded documentation and tutorials.

## Citation and References

Please cite the following report if you use this repository in your research:

**INDOT Report**: [Field-Validated Deep Learning Model for Piezoelectric-Based In-Situ Concrete Strength Sensing](https://rosap.ntl.bts.gov/view/dot/54753)

## License

All rights reserved.

This repository, including all data, code, and associated materials, is the intellectual property of the authors and affiliated institutions. Unauthorized use, distribution, or modification of this repository or any part of its contents is strictly prohibited without explicit written permission from the authors.

### Restrictions
- **Academic Use Only**: The contents of this repository are provided solely for academic and non-commercial research purposes.
- **Prohibited Commercial Use**: Any commercial use, including but not limited to product development, sales, or integration into commercial software, is strictly forbidden.
- **Redistribution**: Redistribution of this repository or its contents, in whole or in part, without prior authorization, is prohibited.
- **Derivative Works**: Modification and creation of derivative works based on this repository are only permitted for academic purposes and must not be used commercially.

### Contact
For inquiries regarding collaboration, licensing, or commercial use, please contact the authors directly.

## Acknowledgments

We thank the Joint Transportation Research Program (JTRP 4210) administered by the Indiana Department of Transportation and Purdue University for their support.
