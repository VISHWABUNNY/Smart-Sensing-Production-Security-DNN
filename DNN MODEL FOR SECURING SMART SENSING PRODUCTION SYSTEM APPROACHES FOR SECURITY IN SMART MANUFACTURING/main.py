from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import pandas as pd
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

main = tkinter.Tk()
main.title("DNN MODEL FOR SECURING SMART SENSING PRODUCTION SYSTEM") 
main.geometry("1600x1300")

global filename
global x_train,y_train,x_test,y_test
global X, Y
global le
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global classifier
global cnn_model

def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global X, y
    global le
    global dataset
       
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")

    le = LabelEncoder()
    dataset['passed_visual_inspection'] = le.fit_transform(dataset['passed_visual_inspection'])
    dataset['machining_finalized'] = le.fit_transform(dataset['machining_finalized'])
    dataset['Machining_Process'] = le.fit_transform(dataset['Machining_Process'])
    X=dataset.iloc[:,0:52]
    y=dataset.iloc[:,-1]
    

    text.insert(END,"Total records found in dataset: "+str(X.shape[0])+"\n\n")
    # Create a count plot
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    # Replace 'dataset' with your actual DataFrame and 'Drug' with the column name
    ax = sns.countplot(x='Machining_Process', data=dataset, palette="Set3")
    plt.title("Count Plot")  # Add a title to the plot
    plt.xlabel("Categories")  # Add label to x-axis
    plt.ylabel("Count")  # Add label to y-axis
    # Annotate each bar with its count value
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()  # Display the plot
def splitting():
    global x_train,y_train,x_test,y_test
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    text.insert(END,"Total records found in dataset to train: "+str(x_train.shape[0])+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(x_test.shape[0])+"\n\n") 

def custom_knn_classifier():
    global x_train, y_train
    text.delete('1.0', END)
    KNN = KNeighborsClassifier(n_neighbors=10,leaf_size=30,metric='minkowski',)  # Create an instance of KNeighborsClassifier
    #x_train_reshaped = np.array(x_train).reshape(-1, 1)
    #x_test_reshaped = np.array(x_test).reshape(-1, 1)
    KNN.fit(x_train, y_train)
    predict = KNN.predict(x_test)
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, "KNN Precision : " + str(p) + "\n")
    text.insert(END, "KNN Recall    : " + str(r) + "\n")
    text.insert(END, "KNN FMeasure  : " + str(f) + "\n")
    text.insert(END, "KNN Accuracy  : " + str(a) + "\n\n")
    # Compute confusion matrix
    cm = confusion_matrix(y_test,predict)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('KNeighborsClassifier Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Compute classification report
    report = classification_report(y_test,predict)
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
def classifier():
    global x_train, y_train, x_test, y_test
    global MLP
    global y_test,model,scaler
    text.delete('1.0', END)
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    y_train1 = y_train.values
    y_test1 = y_test.values

    model_path = 'model.h5'

    # Check if the model file exists
    if os.path.exists(model_path):
        # Load the pre-trained model
        model = load_model(model_path)
    else:
        # Build a simple Deep Neural Network
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(9, activation='softmax')  # Assuming 9 classes for Failure types
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_scaled, y_train1, epochs=10, batch_size=16, validation_split=0.1)

        # Save the trained model
        model.save(model_path)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test1)
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')

    # Predictions and Metrics
    predict = model.predict(X_test_scaled)
    predict = np.argmax(predict, axis=1)
    
    p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=0) * 100
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    # Display precision, recall, F1-score, and accuracy in the Text widget
    text.insert(END, "DNN Algorithm Precision: " + str(p) + "\n")
    text.insert(END, "DNN Algorithm Recall: " + str(r) + "\n")
    text.insert(END, "DNN Algorithm FMeasure: " + str(f) + "\n")
    text.insert(END, "DNN Algorithm Accuracy: " + str(a) + "\n\n")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict)
    
    # Compute classification report
    report = classification_report(y_test, predict)
    
    # Display confusion matrix in the Text widget
    text.insert(END, "Confusion Matrix:\n")
    text.insert(END, str(cm) + "\n\n")
    
    # Display classification report in the Text widget
    text.insert(END, "Classification Report:\n")
    text.insert(END, report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('DNN  Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    report= classification_report(y_test, predict)
    print(report)

def predict():
    global label_encoder, labels, columns, types, pca, scaler, model,le
    # Assuming 'model' is your pre-trained model and 'text' is your tkinter text widget
    text.delete('1.0', END)
    
    filename = filedialog.askopenfilename(initialdir="testData")
    test = pd.read_csv(filename)
    test['passed_visual_inspection'] = le.fit_transform(test['passed_visual_inspection'])
    test['machining_finalized'] = le.fit_transform(test['machining_finalized'])
    # Assuming 'scaler' is the scaler used during training
    test_scaled = scaler.transform(test)
    
    predictions = []
    
    for i in range(len(test)):
        # Convert the Pandas DataFrame to a NumPy array
        row = test_scaled[i].reshape(1, -1)
        
        # Assuming 'model' is your pre-trained model
        predicted_data = model.predict(row)
        
        # Extract the predicted class index
        predicted_class = np.argmax(predicted_data)
        
        
        # Map the class index to the corresponding label
        if predicted_class == 0:
            predicted_label = "End"
        elif predicted_class == 1:
            predicted_label = "Layer 1 up"
        elif predicted_class == 2:
            predicted_label = "Layer 1 Down"
        elif predicted_class == 3:
            predicted_label = "Layer 2 Down"
        elif predicted_class == 4:
            predicted_label = "Layer 2 Up"
        elif predicted_class == 5:
            predicted_label = "Layer 3 Up"
        elif predicted_class == 6:
            predicted_label = "Layer 3 Down"
        elif predicted_class == 7:
            predicted_label = "Prep"
        elif predicted_class == 8:
            predicted_label = "Repositioning"
        
        predictions.append(predicted_label)
        
        text.insert(END, f'Test for row {i}: {row}\n')
        text.insert(END, f'Predicted output for row {i}:***********************************{predicted_label}\n')
    return predictions
def graph():
    # Create a DataFrame
    df = pd.DataFrame([
    ['KNN', 'Precision', precision[0]],
    ['KNN', 'Recall', recall[0]],
    ['KNN', 'F1 Score', fscore[0]],
    ['KNN', 'Accuracy', accuracy[0]],
    ['DNN', 'Precision', precision[-1]],
    ['DNN', 'Recall', recall[-1]],
    ['DNN', 'F1 Score', fscore[-1]],
    ['DNN', 'Accuracy', accuracy[-1]],
    ], columns=['Parameters', 'Algorithms', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Parameters', columns='Algorithms', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    # Set graph properties
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='DNN MODEL FOR SECURING SMART SENSING PRODUCTION SYSTEM', justify=LEFT)
title.config(bg='gray', fg='blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=200,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=500,y=100)
preprocessButton.config(font=font1)

analysisButton = Button(main, text="Data splitting", command=splitting)
analysisButton.place(x=200,y=150)
analysisButton.config(font=font1) 

knnButton = Button(main, text="KNeighborsClassifier", command=custom_knn_classifier)
knnButton.place(x=500,y=150)
knnButton.config(font=font1)

LRButton = Button(main, text="DNN Algorithm", command=classifier)
LRButton.place(x=200,y=200)
LRButton.config(font=font1)

predictButton = Button(main, text="Prediction", command=predict)
predictButton.place(x=500,y=200)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=200,y=250)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=500,y=250)
exitButton.config(font=font1)

                            

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1) 

main.config(bg='LightSteelBlue1')
main.mainloop()






