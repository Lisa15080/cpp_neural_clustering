import sys
import csv
import random
import math
from pathlib import Path
import os

os.system("chcp 65001 > nul")
MODULE_DIR = r"C:\Users\annys\CLionProjects\cpp_neural_clustering\cmake-build-debug"
MINGW_DIR = r"C:\Program Files\JetBrains\CLion 2025.2.5\bin\mingw\bin"

sys.path.insert(0, MODULE_DIR)

os.add_dll_directory(MODULE_DIR)
os.add_dll_directory(MINGW_DIR)

import cpp_neural_clustering as cnc

print("Модуль загружен:", cnc.__doc__)

import cpp_neural_clustering as cnc


# ============================================================
# 2. Путь к датасету
# ============================================================

DATASET_PATH = Path(__file__).resolve().parent / "dataset2 (1).csv"


# ============================================================
# 3. Загрузка CSV
# ============================================================

def load_dataset(path):
    X = []
    y = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            x1 = float(row["feature_0"])
            x2 = float(row["feature_1"])
            target = float(row["target"])

            X.append([x1, x2])
            y.append([target])

    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    data = list(zip(X, y))

    random.seed(seed)
    random.shuffle(data)

    split_index = int(len(data) * (1.0 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]

    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    return X_train, X_test, y_train, y_test


def compute_mean_std(X):
    n_features = len(X[0])

    means = []
    stds = []

    for j in range(n_features):
        values = [row[j] for row in X]

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)

        if std == 0:
            std = 1.0

        means.append(mean)
        stds.append(std)

    return means, stds


def normalize(X, means, stds):
    normalized = []

    for row in X:
        new_row = []

        for j in range(len(row)):
            new_value = (row[j] - means[j]) / stds[j]
            new_row.append(new_value)

        normalized.append(new_row)

    return normalized


# ============================================================
# 6. Метрики
# ============================================================

def predict_binary(trainer, x, threshold=0.5):
    probability = trainer.predict(x)[0]

    if probability >= threshold:
        return 1, probability

    return 0, probability


def calculate_metrics(trainer, X, y):
    correct = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x_item, y_item in zip(X, y):
        actual = int(y_item[0])
        predicted, probability = predict_binary(trainer, x_item)

        if predicted == actual:
            correct += 1

        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 1:
            fn += 1

    accuracy = correct / len(y)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }



def main():
    print("Модуль загружен:", cnc.__doc__)

    X, y = load_dataset(DATASET_PATH)

    print("Всего объектов:", len(X))
    print("Размер входа:", len(X[0]))
    print("Пример X:", X[0])
    print("Пример y:", y[0])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        seed=42
    )

    means, stds = compute_mean_std(X_train)

    X_train = normalize(X_train, means, stds)
    X_test = normalize(X_test, means, stds)

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # Архитектура:
    # 2 входа -> 8 нейронов -> 4 нейрона -> 1 выход
    network = cnc.NeuralNetwork(
        [2, 8, 4, 1],
        cnc.Activation.RELU,
        False,
        ""
    )

    config = cnc.TrainingConfig()
    config.epochs = 1000
    config.learning_rate = 0.05
    config.verbose = True

    trainer = cnc.Trainer(network, config)

    print("\nНачинаем обучение...")
    trainer.train(X_train, y_train)

    print("\nОбучение завершено.")

    # Встроенная evaluate из C++ возвращает accuracy в процентах
    train_accuracy_cpp = trainer.evaluate(X_train, y_train)
    test_accuracy_cpp = trainer.evaluate(X_test, y_test)

    print("\nAccuracy через C++ evaluate:")
    print(f"Train accuracy: {train_accuracy_cpp:.2f}%")
    print(f"Test accuracy:  {test_accuracy_cpp:.2f}%")

    train_metrics = calculate_metrics(trainer, X_train, y_train)
    test_metrics = calculate_metrics(trainer, X_test, y_test)

    print("\nМетрики на train:")
    print(f"Accuracy:  {train_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall:    {train_metrics['recall']:.4f}")
    print(f"F1-score:  {train_metrics['f1']:.4f}")
    print(f"TP: {train_metrics['tp']}, TN: {train_metrics['tn']}, FP: {train_metrics['fp']}, FN: {train_metrics['fn']}")

    print("\nМетрики на test:")
    print(f"Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1']:.4f}")
    print(f"TP: {test_metrics['tp']}, TN: {test_metrics['tn']}, FP: {test_metrics['fp']}, FN: {test_metrics['fn']}")

    print("\nПервые 10 предсказаний на test:")
    for i in range(min(10, len(X_test))):
        predicted, probability = predict_binary(trainer, X_test[i])
        actual = int(y_test[i][0])

        print(
            f"{i + 1}) actual={actual}, "
            f"predicted={predicted}, "
            f"probability={probability:.4f}"
        )


if __name__ == "__main__":
    main()