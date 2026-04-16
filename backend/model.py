import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')
TARGET_LABELS = {
    0: 'End',
    1: 'Layer 1 Up',
    2: 'Layer 1 Down',
    3: 'Layer 2 Down',
    4: 'Layer 2 Up',
    5: 'Layer 3 Up',
    6: 'Layer 3 Down',
    7: 'Prep',
    8: 'Repositioning',
}


def load_dataset(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError('Dataset path is required')
    return pd.read_csv(path)


def find_target_column(df: pd.DataFrame) -> str:
    for col in ['Machining_Process', 'machining_process', 'target', 'label']:
        if col in df.columns:
            return col
    raise KeyError(
        'Target column not found. Expected one of: Machining_Process, machining_process, target, label.'
    )


def preprocess_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], str]:
    df = dataset.copy()
    df.fillna(0, inplace=True)

    target_col = find_target_column(df)

    encoders = {
        'passed_visual_inspection': LabelEncoder(),
        'machining_finalized': LabelEncoder(),
        'target': LabelEncoder(),
    }

    if 'passed_visual_inspection' not in df.columns or 'machining_finalized' not in df.columns:
        missing = [col for col in ['passed_visual_inspection', 'machining_finalized'] if col not in df.columns]
        raise KeyError(f'Missing required columns: {missing}')

    df['passed_visual_inspection'] = encoders['passed_visual_inspection'].fit_transform(
        df['passed_visual_inspection'].astype(str)
    )
    df['machining_finalized'] = encoders['machining_finalized'].fit_transform(
        df['machining_finalized'].astype(str)
    )
    df[target_col] = encoders['target'].fit_transform(
        df[target_col].astype(str)
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, encoders, target_col


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'accuracy': accuracy_score(y_true, y_pred) * 100,
    }


def train_knn_classifier(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, n_neighbors: int = 10) -> Dict:
    model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=30, metric='minkowski')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results = evaluate_predictions(y_test, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    results['classification_report'] = classification_report(y_test, y_pred)
    return {
        'model': model,
        'predictions': y_pred,
        'results': results,
    }


def build_dnn_model(input_dim: int, num_classes: int) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_or_load_dnn(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, model_path: str = MODEL_PATH, epochs: int = 10, batch_size: int = 16) -> Dict:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            model.predict(x_train_scaled[:1])
        except Exception:
            model = None

    if model is None:
        num_classes = len(np.unique(y_train))
        model = build_dnn_model(input_dim=x_train_scaled.shape[1], num_classes=num_classes)
        model.fit(x_train_scaled, y_train.values, epochs=epochs, batch_size=batch_size, validation_split=0.1)
        model.save(model_path)

    test_loss, test_acc = model.evaluate(x_test_scaled, y_test.values, verbose=0)
    y_pred = np.argmax(model.predict(x_test_scaled), axis=1)

    results = evaluate_predictions(y_test, y_pred)
    results['test_loss'] = float(test_loss)
    results['test_accuracy'] = float(test_acc)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    results['classification_report'] = classification_report(y_test, y_pred)

    return {
        'model': model,
        'scaler': scaler,
        'predictions': y_pred,
        'results': results,
    }


def predict_with_model(model: models.Model, scaler: StandardScaler, encoders: Dict[str, LabelEncoder], dataset: pd.DataFrame) -> List[str]:
    df = dataset.copy()
    df.fillna(0, inplace=True)

    if 'passed_visual_inspection' in df.columns:
        df['passed_visual_inspection'] = encoders['passed_visual_inspection'].transform(df['passed_visual_inspection'].astype(str))
    if 'machining_finalized' in df.columns:
        df['machining_finalized'] = encoders['machining_finalized'].transform(df['machining_finalized'].astype(str))

    if 'Machining_Process' in df.columns:
        columns = [c for c in df.columns if c != 'Machining_Process']
    else:
        columns = df.columns.tolist()

    features = df.loc[:, columns]
    features_scaled = scaler.transform(features)

    raw_predictions = model.predict(features_scaled)
    predicted_indices = np.argmax(raw_predictions, axis=1)
    return encoders['target'].inverse_transform(predicted_indices)


def plot_comparison(knn_results: Dict, dnn_results: Dict) -> None:
    if not knn_results or not dnn_results:
        raise ValueError('Both KNN and DNN results are required for comparison')

    data = [
        ['Precision', 'KNN', knn_results['results']['precision']],
        ['Recall', 'KNN', knn_results['results']['recall']],
        ['F1 Score', 'KNN', knn_results['results']['f1_score']],
        ['Accuracy', 'KNN', knn_results['results']['accuracy']],
        ['Precision', 'DNN', dnn_results['results']['precision']],
        ['Recall', 'DNN', dnn_results['results']['recall']],
        ['F1 Score', 'DNN', dnn_results['results']['f1_score']],
        ['Accuracy', 'DNN', dnn_results['results']['accuracy']],
    ]
    df = pd.DataFrame(data, columns=['Metric', 'Model', 'Value'])
    pivot_df = df.pivot_table(index='Metric', columns='Model', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar')
    plt.title('Classifier Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
