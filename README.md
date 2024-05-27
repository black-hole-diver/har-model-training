# Human Activity Recognition Model Training
This repository contains all the codes used for data cleaning, data processing, model training, and model deployment for the thesis "Machine Learning Based Real-time Movement Detection of Children" (BSc Thesis, 2024, ELTE IK).

## Dataset Used
The dataset utilized in this project is derived from a study that involved 40 children, with the goal of recognizing various activities through wearable devices. The data, structured in pickle files and accompanying CSV files for labels, provide a comprehensive basis for training our machine learning models.

### Data Format

The dataset includes:
- **Pickle Files**: Contain raw sensor data from wearable devices, capturing movement specifics of each child involved in the study.
- **CSV Files**: Accompany the pickle files and contain labels for the movements. Each label corresponds to a specific activity recorded in the pickle files.

### Source

The methodology and detailed description of how the data was collected can be found in the following research paper:
- [Human Activity Recognition of Children with Wearable Devices Using LightGBM Machine Learning](https://www.researchgate.net/publication/356867052_Human_Activity_Recognition_of_Children_with_Wearable_Devices_Using_LightGBM_Machine_Learning)

## Setup and Installation

To set up the project environment, follow these 2 steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository-url
   cd your-repository-directory

2. **Install Dependencies:**
  Ensure you have Python installed, then run:
  ```
  pip install -r requirements.txt
  ```
