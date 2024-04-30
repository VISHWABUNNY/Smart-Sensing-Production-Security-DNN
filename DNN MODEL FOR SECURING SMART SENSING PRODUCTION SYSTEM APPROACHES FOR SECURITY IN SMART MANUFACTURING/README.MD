DNN Model for Securing Smart Sensing Production System



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



Dependencies:

Python 3.10
tkinter
numpy
pandas
seaborn
scikit-learn
imbalanced-learn
matplotlib
tensorflow



Usage:

Clone the repository.
Install the dependencies using pip install -r requirements.txt.
Run main.py to launch the application.
Follow the on-screen instructions to upload, preprocess, train, evaluate, and make predictions using the dataset.



Dataset:

The dataset used in this project should be in CSV format and contain sensor data related to the machining process. It should include features such as temperature, pressure, vibration, etc., along with the corresponding labels indicating the stage of the machining process.