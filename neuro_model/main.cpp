#include "Neural_Net/neural_net.h"
#include "../Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"
#include "../DataSet/dataset.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include "libs/json.hpp"

#ifdef _WIN32
    #include <direct.h>
#endif

using namespace std;

// Получить текущую рабочую директорию
string getCurrentPath() {
    char buffer[1024];
#ifdef _WIN32
    if (_getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#endif
    return ".";
}

// Преобразовать Matrix<double> в vector<vector<double>>
vector<vector<double>> matrixToVector(const Matrix<double>& mat) {
    vector<vector<double>> vec(mat.rows(), vector<double>(mat.cols()));
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Преобразовать vector<vector<double>> в Matrix<double>
Matrix<double> vectorToMatrix(const vector<vector<double>>& vec) {
    if (vec.empty()) return Matrix<double>();
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Matrix<double> mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = vec[i][j];
    return mat;
}

// Нормализация данных
void normalizeData(vector<vector<double>>& inputs) {
    if (inputs.empty() || inputs[0].empty()) return;

    size_t n_samples = inputs.size();
    size_t n_features = inputs[0].size();

    vector<double> mean(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            mean[j] += inputs[i][j];
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        mean[j] /= n_samples;
    }

    vector<double> std(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            std[j] += (inputs[i][j] - mean[j]) * (inputs[i][j] - mean[j]);
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        std[j] = sqrt(std[j] / n_samples);
        if (std[j] < 1e-8) std[j] = 1.0;
    }

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            inputs[i][j] = (inputs[i][j] - mean[j]) / std[j];
        }
    }
}

// Функция для разделения данных на train и validation
void trainValidationSplit(const vector<vector<double>>& features,
                          const vector<vector<double>>& targets,
                          vector<vector<double>>& train_features,
                          vector<vector<double>>& train_targets,
                          vector<vector<double>>& val_features,
                          vector<vector<double>>& val_targets,
                          double val_ratio = 0.2) {
    size_t total = features.size();
    size_t val_size = static_cast<size_t>(total * val_ratio);
    size_t train_size = total - val_size;

    train_features.assign(features.begin(), features.begin() + train_size);
    train_targets.assign(targets.begin(), targets.begin() + train_size);
    val_features.assign(features.begin() + train_size, features.end());
    val_targets.assign(targets.begin() + train_size, targets.end());
}

void saveToJSON(const vector<vector<double>>& inputs,
                const vector<vector<double>>& targets,
                const string& filename) {
    nlohmann::json data;
    vector<double> x, y, labels;

    for (size_t i = 0; i < inputs.size(); ++i) {
        x.push_back(inputs[i][0]);
        y.push_back(inputs[i][1]);
        labels.push_back(targets[i][0]);
    }

    data["x"] = x;
    data["y"] = y;
    data["labels"] = labels;

    ofstream file(filename);
    file << data.dump(4);
    file.close();
}

// Главная программа
int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    cout << "=== Нейронная сеть - бинарная классификация (Синтетические данные) ===\n";
    cout << "Текущая папка: " << getCurrentPath() << "\n\n";

    // Объявляем переменные ДО try, чтобы они были доступны после
    Dataset synthetic_data;
    Trainer* trainer_ptr = nullptr;
    NeuralNetwork* net_ptr = nullptr;
    TrainingConfig cfg;

    try {
        cout << "[1] Генерация синтетического датасета...\n";

        // Параметры датасета
        size_t n_samples = 10000;      // 10000 точек
        double cluster_std = 0.8;      // стандартное отклонение кластеров
        double separation = 3.0;       // расстояние между центрами кластеров

        // Генерируем данные с помощью DatasetGenerator
        DatasetGenerator generator;
        synthetic_data = generator.generate_gaussian(n_samples, cluster_std, separation);

        cout << "  - Сгенерировано примеров: " << synthetic_data.inputs.size() << "\n";
        cout << "  - Признаков: " << synthetic_data.inputs[0].size() << " (x, y)\n";
        cout << "  - Расстояние между центрами кластеров: " << separation << "\n";
        cout << "  - Стандартное отклонение: " << cluster_std << "\n";

        // Подсчет классов
        int count0 = 0, count1 = 0;
        for (const auto& t : synthetic_data.targets) {
            if (t[0] < 0.5) count0++;
            else count1++;
        }
        cout << "  - Класс 0: " << count0 << ", класс 1: " << count1 << "\n";
        cout << "  - Дисбаланс: " << fixed << setprecision(2)
             << (double)count1 / count0 << ":1\n";

        // Нормализация данных
        cout << "\n[2] Нормализация данных...\n";
        normalizeData(synthetic_data.inputs);
        cout << "  - Нормализация завершена\n";

        // Разделение на train (80%) и validation (20%)
        vector<vector<double>> train_features, train_targets;
        vector<vector<double>> val_features, val_targets;

        trainValidationSplit(synthetic_data.inputs, synthetic_data.targets,
                            train_features, train_targets,
                            val_features, val_targets, 0.2);

        cout << "\n[3] Разделение данных:\n";
        cout << "  - Train samples: " << train_features.size() << "\n";
        cout << "  - Validation samples: " << val_features.size() << "\n";

        // Создание сети
        size_t n_features = train_features[0].size();
        cout << "\n[4] Создание сети (архитектура: " << n_features << " -> 32 -> 16 -> 1)\n";
        NeuralNetwork net({(int)n_features, 32, 16, 1}, Activation::RELU, true, "log.txt");
        net_ptr = &net;

        // Обучение
        cout << "\n[5] Обучение...\n";
        cfg.epochs = 300;
        cfg.learning_rate = 0.01;
        cfg.verbose = true;

        Trainer trainer(net, cfg);
        trainer_ptr = &trainer;
        trainer.train(train_features, train_targets);

        // Оценка на валидационных данных
        Matrix<double> val_inputs_mat = vectorToMatrix(val_features);
        Matrix<double> val_targets_mat = vectorToMatrix(val_targets);
        double val_accuracy = trainer.evaluate(val_inputs_mat, val_targets_mat);
        cout << "\n[6] Точность на валидационных данных: " << fixed << setprecision(2)
             << val_accuracy << "%\n";

        // Оценка на обучающих данных
        Matrix<double> train_inputs_mat = vectorToMatrix(train_features);
        Matrix<double> train_targets_mat = vectorToMatrix(train_targets);
        double train_accuracy = trainer.evaluate(train_inputs_mat, train_targets_mat);
        cout << "  - Точность на обучающих данных: " << fixed << setprecision(2)
             << train_accuracy << "%\n";

        // Визуализация разделяющей границы (тестовые точки)
        cout << "\n[7] Примеры предсказаний (тестовые точки):\n";

        // Создаем сетку точек для демонстрации
        vector<vector<double>> test_points = {
            {2.0, 0.0},   // около центра класса 0
            {-2.0, 0.0},  // около центра класса 1
            {0.0, 0.0},   // граница
            {1.5, 1.5},   // дальняя точка
            {-1.5, -1.5}  // дальняя точка
        };

        for (const auto& pt : test_points) {
            vector<double> out = trainer.predict(pt);
            int pred = (out[0] > 0.5) ? 1 : 0;
            cout << "  Точка (" << fixed << setprecision(2) << pt[0] << ", " << pt[1]
                 << ") -> P(1) = " << setprecision(4) << out[0]
                 << ", класс: " << pred << "\n";
        }

        // Сохранение модели
        net.saveModel("synthetic_model.txt");
        cout << "\n[8] Модель сохранена в synthetic_model.txt\n";

        // Сохранение данных в CSV для визуализации
        ofstream data_file("synthetic_data.csv");
        data_file << "x,y,class\n";
        for (size_t i = 0; i < synthetic_data.inputs.size(); ++i) {
            data_file << synthetic_data.inputs[i][0] << ","
                      << synthetic_data.inputs[i][1] << ","
                      << synthetic_data.targets[i][0] << "\n";
        }
        data_file.close();
        cout << "  - Данные сохранены в synthetic_data.csv (для визуализации)\n";

        // Сохранение JSON для Python визуализации
        cout << "\n[9] Сохранение данных для визуализации...\n";
        saveToJSON(synthetic_data.inputs, synthetic_data.targets, "true_clusters.json");

        // Для предсказаний
        vector<vector<double>> predictions;
        for (size_t i = 0; i < synthetic_data.inputs.size(); ++i) {
            vector<double> out = trainer.predict(synthetic_data.inputs[i]);
            predictions.push_back({(out[0] > 0.5) ? 1.0 : 0.0});
        }
        saveToJSON(synthetic_data.inputs, predictions, "predictions.json");

        // Итоговая статистика
        cout << "\n=== Итоговая статистика ===\n";
        cout << "  - Архитектура сети: " << n_features << " → 32 → 16 → 1\n";
        cout << "  - Эпохи: " << cfg.epochs << "\n";
        cout << "  - Learning rate: " << cfg.learning_rate << "\n";
        cout << "  - Точность на валидации: " << val_accuracy << "%\n";
        cout << "  - Файлы созданы: synthetic_model.txt, log.txt, synthetic_data.csv, true_clusters.json, predictions.json\n";

        // Запуск Python скрипта
        string copy_cmd1 = "copy true_clusters.json ..\\neuro_model\\";
        string copy_cmd2 = "copy predictions.json ..\\neuro_model\\";

        int copy1 = system(copy_cmd1.c_str());
        int copy2 = system(copy_cmd2.c_str());
        cout << "\n[10] Запуск Python скрипта для визуализации...\n";
        int result = system("py ../neuro_model/plot.py");
        if (result == 0) {
            cout << "  ✅ Визуализация завершена. Результат: result.png\n";
        } else {
            cout << "  ⚠️ Не удалось запустить Python. Убедитесь, что Python установлен.\n";
        }

    } catch (const exception& e) {
        cerr << "\n❌ Ошибка: " << e.what() << "\n";
        return 1;
    }

    cout << "\n=== Программа успешно завершена ===\n";
    return 0;
}