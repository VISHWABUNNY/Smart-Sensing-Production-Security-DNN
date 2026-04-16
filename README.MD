DNN Model for Securing Smart Sensing Production System

Project structure:
- `backend/` — model training, preprocessing, prediction logic
- `frontend/` — Tkinter GUI code
- `Dataset/` — CSV data files
- `main.py` — root entrypoint to launch the GUI
- `run_project.sh` — creates separate virtual environments and runs the project

Overview:

This project implements a Deep Neural Network (DNN) model for securing smart sensing production systems in the context of smart manufacturing. The model is designed to classify various stages of the machining process based on sensor data.



Features:

Upload Dataset: Upload the dataset containing sensor data for the machining process.
Preprocess Dataset: Preprocess the uploaded dataset, including handling missing values and encoding categorical variables.
Data Splitting: Split the dataset into training and testing sets.
KNeighborsClassifier: Train and evaluate a K Nearest Neighbors (KNN) classifier for the classification task.
DNN Algorithm: Train and evaluate a Deep Neural Network (DNN) model for the classification task.
Prediction: Make predictions using the trained models on new data.
Comparison Graph: Visualize the performance comparison between the KNN classifier and the DNN model.
Exit: Close the application.


Use Cases:

- Quality assurance: classify machining stages and detect process deviations using real-time sensor signals.
- Production monitoring: identify abnormal operation states such as layer misalignment, repositioning, or prep stages.
- Predictive maintenance: flag suspicious machining behavior before failures occur using historical sensor patterns.
- Security in smart manufacturing: support anomaly detection in automated production lines by monitoring endpoint classification drift.

What it accomplishes:

- Converts raw sensor streams into labeled machining stage predictions.
- Trains and compares two model families (KNN and DNN) from the same dataset.
- Provides a GUI for non-technical users to upload data, preprocess it, train models, and inspect outputs.
- Saves a reusable DNN model so predictions can be made quickly on new production data.

Value delivered:

- Faster decision support for manufacturing engineers by turning complex sensor data into actionable classifications.
- Improved process visibility through model performance metrics and comparison charts.
- Reduced manual inspection effort with automated defect and stage prediction.
- A practical starting point for smart factory security workflows that need real-time classification of machining states.


Dependencies:

- `backend/requirements.txt`
- `frontend/requirements.txt`

Usage:

- Run `./run_project.sh` from the repository root.
- The script creates separate virtual environments at `backend/.venv` and `frontend/.venv`, installs each side's dependencies there, and then launches the app.



Dataset:

The dataset used in this project should be in CSV format and contain sensor data related to the machining process. It should include features such as temperature, pressure, vibration, etc., along with the corresponding labels indicating the stage of the machining process.