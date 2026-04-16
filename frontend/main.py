import os
import tkinter as tk
from tkinter import END, LEFT, Button, Label, Scrollbar, Text, filedialog

from backend.model import (
    load_dataset,
    plot_comparison,
    predict_with_model,
    preprocess_dataset as backend_preprocess_dataset,
    split_data,
    train_knn_classifier,
    train_or_load_dnn,
)

root_dir = os.getcwd()

dataset = None
X = None
y = None
encoders = None
target_col = None
x_train = None
x_test = None
y_train = None
y_test = None
knn_results = None

dnn_model = None
scaler = None
dnn_results = None


def upload_dataset():
    global dataset
    file_path = filedialog.askopenfilename(
        initialdir=os.path.join(root_dir, 'Dataset'),
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    )
    if not file_path:
        return

    dataset = load_dataset(file_path)
    text.delete('1.0', END)
    text.insert(END, str(dataset.head()) + '\n\n')


def preprocess_dataset():
    global X, y, encoders, target_col
    if dataset is None:
        text.delete('1.0', END)
        text.insert(END, 'Please upload a dataset first.\n')
        return

    X, y, encoders, target_col = backend_preprocess_dataset(dataset)
    text.delete('1.0', END)
    text.insert(END, str(dataset.head()) + '\n\n')
    text.insert(END, f'Total records found in dataset: {X.shape[0]}\n\n')


def splitting():
    global x_train, x_test, y_train, y_test
    if X is None or y is None:
        text.delete('1.0', END)
        text.insert(END, 'Please preprocess the dataset first.\n')
        return

    x_train, x_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
    text.delete('1.0', END)
    text.insert(END, f'Total records found in dataset to train: {x_train.shape[0]}\n')
    text.insert(END, f'Total records found in dataset to test: {x_test.shape[0]}\n')


def custom_knn_classifier():
    global knn_results
    if x_train is None or y_train is None or x_test is None or y_test is None:
        text.delete('1.0', END)
        text.insert(END, 'Please split the dataset before training KNN.\n')
        return

    knn_results = train_knn_classifier(x_train, y_train, x_test, y_test)
    results = knn_results['results']

    text.delete('1.0', END)
    text.insert(END, f"KNN Precision : {results['precision']:.2f}\n")
    text.insert(END, f"KNN Recall    : {results['recall']:.2f}\n")
    text.insert(END, f"KNN F1 Score  : {results['f1_score']:.2f}\n")
    text.insert(END, f"KNN Accuracy  : {results['accuracy']:.2f}\n\n")
    text.insert(END, 'Confusion Matrix:\n')
    text.insert(END, str(results['confusion_matrix']) + '\n\n')
    text.insert(END, 'Classification Report:\n')
    text.insert(END, results['classification_report'])


def classifier():
    global dnn_model, scaler, dnn_results
    if x_train is None or y_train is None or x_test is None or y_test is None:
        text.delete('1.0', END)
        text.insert(END, 'Please split the dataset before training the DNN.\n')
        return

    dnn_results = train_or_load_dnn(x_train, y_train, x_test, y_test)
    dnn_model = dnn_results['model']
    scaler = dnn_results['scaler']
    results = dnn_results['results']

    text.delete('1.0', END)
    text.insert(END, f"DNN Precision: {results['precision']:.2f}\n")
    text.insert(END, f"DNN Recall   : {results['recall']:.2f}\n")
    text.insert(END, f"DNN F1 Score : {results['f1_score']:.2f}\n")
    text.insert(END, f"DNN Accuracy : {results['accuracy']:.2f}\n\n")
    text.insert(END, 'Confusion Matrix:\n')
    text.insert(END, str(results['confusion_matrix']) + '\n\n')
    text.insert(END, 'Classification Report:\n')
    text.insert(END, results['classification_report'])


def predict():
    if dnn_model is None or scaler is None or encoders is None:
        text.delete('1.0', END)
        text.insert(END, 'Please train or load the DNN model before prediction.\n')
        return

    file_path = filedialog.askopenfilename(
        initialdir=os.path.join(root_dir, 'Dataset'),
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    )
    if not file_path:
        return

    test_data = load_dataset(file_path)
    predictions = predict_with_model(dnn_model, scaler, encoders, test_data)

    text.delete('1.0', END)
    for index, label in enumerate(predictions):
        text.insert(END, f'Row {index}: {label}\n')


def graph():
    if knn_results is None or dnn_results is None:
        text.delete('1.0', END)
        text.insert(END, 'Please run both KNN and DNN evaluations before showing the comparison graph.\n')
        return

    plot_comparison(knn_results, dnn_results)


def close():
    main.destroy()


def run_app():
    global main, text
    main = tk.Tk()
    main.title('DNN Production Security Dashboard')
    main.geometry('1400x900')
    main.configure(bg='#f5f7fb')

    header_frame = tk.Frame(main, bg='#273c75', padx=20, pady=20)
    header_frame.pack(fill='x')
    title = tk.Label(header_frame, text='Smart Sensing Production Security', fg='white', bg='#273c75', font=('Helvetica', 24, 'bold'))
    subtitle = tk.Label(header_frame, text='Visualize, train and predict machining process security states', fg='white', bg='#273c75', font=('Helvetica', 12))
    title.pack(anchor='w')
    subtitle.pack(anchor='w', pady=(5, 0))

    body_frame = tk.Frame(main, bg='#f5f7fb', padx=20, pady=20)
    body_frame.pack(fill='both', expand=True)

    control_frame = tk.Frame(body_frame, bg='white', bd=2, relief='groove', padx=20, pady=20)
    control_frame.pack(side='left', fill='y', ipadx=10, ipady=10)

    output_frame = tk.Frame(body_frame, bg='white', bd=2, relief='groove')
    output_frame.pack(side='right', fill='both', expand=True, padx=(20, 0))

    control_label = tk.Label(control_frame, text='Controls', bg='white', fg='#192a56', font=('Helvetica', 16, 'bold'))
    control_label.pack(anchor='w', pady=(0, 10))

    button_style = {'font': ('Helvetica', 12, 'bold'), 'bg': '#192a56', 'fg': 'white', 'activebackground': '#40739e', 'activeforeground': 'white', 'bd': 0, 'padx': 18, 'pady': 10}

    tk.Button(control_frame, text='Upload Dataset', command=upload_dataset, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Preprocess Dataset', command=preprocess_dataset, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Split Data', command=splitting, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Train KNN', command=custom_knn_classifier, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Train DNN', command=classifier, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Predict', command=predict, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Show Comparison', command=graph, **button_style).pack(fill='x', pady=6)
    tk.Button(control_frame, text='Exit App', command=close, **button_style).pack(fill='x', pady=6)

    status_frame = tk.Frame(control_frame, bg='#f7f9fc')
    status_frame.pack(fill='x', pady=(20, 0))
    status_title = tk.Label(status_frame, text='Status', bg='#f7f9fc', fg='#192a56', font=('Helvetica', 14, 'bold'))
    status_title.pack(anchor='w', pady=(0, 6))
    status_text = tk.Label(status_frame, text='Ready to start.', bg='#f7f9fc', fg='#30336b', wraplength=240, justify='left')
    status_text.pack(anchor='w')

    output_label = tk.Label(output_frame, text='Application Output', bg='white', fg='#192a56', font=('Helvetica', 16, 'bold'), padx=20, pady=20)
    output_label.pack(anchor='w')

    text_frame = tk.Frame(output_frame, bg='white', padx=20, pady=10)
    text_frame.pack(fill='both', expand=True)

    text = Text(text_frame, wrap='word', font=('Helvetica', 11), bd=0, bg='#f5f7fb', fg='#2f3640')
    scroll = Scrollbar(text_frame, command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right', fill='y')
    text.pack(side='left', fill='both', expand=True)

    main.mainloop()


if __name__ == '__main__':
    run_app()
