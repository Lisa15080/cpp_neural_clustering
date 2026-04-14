#include "Neural_Net/neural_net.h"
#include "Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"
#include "../parser/pars.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib> 

#ifdef _WIN32
    #include <direct.h>
    #define getcwd _getcwd
#else
    #include <unistd.h>
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

// Нормализация данных (с сохранением параметров для тестовой выборки)
struct NormalizationParams {
    vector<double> mean;
    vector<double> std;
};

NormalizationParams fitNormalizeData(vector<vector<double>>& inputs) {
    NormalizationParams params;
    if (inputs.empty() || inputs[0].empty()) return params;

    size_t n_samples = inputs.size();
    size_t n_features = inputs[0].size();

    params.mean.assign(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            params.mean[j] += inputs[i][j];
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        params.mean[j] /= n_samples;
    }

    params.std.assign(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            params.std[j] += (inputs[i][j] - params.mean[j]) * (inputs[i][j] - params.mean[j]);
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        params.std[j] = sqrt(params.std[j] / n_samples);
        if (params.std[j] < 1e-8) params.std[j] = 1.0;
    }

    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];
        }
    }

    return params;
}

void normalizeDataWithParams(vector<vector<double>>& inputs, const NormalizationParams& params) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];
        }
    }
}

// Преобразование категориальных признаков в числа
vector<double> processCategoricalRow(const vector<string>& row, const vector<string>& headers) {
    vector<double> features;

    static const map<string, int> geography_map = {{"France", 0}, {"Spain", 1}, {"Germany", 2}};
    static const map<string, int> gender_map = {{"Female", 0}, {"Male", 1}};

    for (size_t i = 0; i < headers.size(); ++i) {
        string header = headers[i];
        string value = row[i];

        if (header == "Geography") {
            auto it = geography_map.find(value);
            features.push_back(it != geography_map.end() ? it->second : 0);
        }
        else if (header == "Gender") {
            auto it = gender_map.find(value);
            features.push_back(it != gender_map.end() ? it->second : 0);
        }
        else if (header == "RowNumber" || header == "CustomerId" || header == "Surname") {
            continue;  // пропускаем
        }
        else if (header == "Exited") {
            continue;  // цель, обработаем отдельно
        }
        else {
            try {
                features.push_back(stod(value));
            } catch (...) {
                features.push_back(0.0);
            }
        }
    }

    return features;
}

// Главная программа
int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    cout << "=== Нейронная сеть - бинарная классификация (Bank Churn) ===\n";
    cout << "Текущая папка: " << getCurrentPath() << "\n\n";

    try {
        cout << "[1] Загрузка данных...\n";

        string train_file = "../Kegal_detaset/train.csv";
        string test_file = "../Kegal_detaset/test.csv";

        // Создаем парсер
        CSVParser parser(',', true);

        // Получаем заголовки
        vector<string> headers = parser.getHeaders(train_file);

        // Читаем train.csv построчно (чтобы обработать категориальные признаки)
        ifstream train_stream(train_file);
        string line;

        // Пропускаем заголовок
        getline(train_stream, line);

        vector<vector<double>> train_features;
        vector<double> train_targets;

        while (getline(train_stream, line)) {
            if (line.empty()) continue;

            stringstream ss(line);
            string token;
            vector<string> row;

            while (getline(ss, token, ',')) {
                row.push_back(token);
            }

            // Обрабатываем признаки
            vector<double> features = processCategoricalRow(row, headers);
            train_features.push_back(features);

            // Находим целевое значение (колонка Exited)
            for (size_t i = 0; i < headers.size(); ++i) {
                if (headers[i] == "Exited" && i < row.size()) {
                    try {
                        train_targets.push_back(stod(row[i]));
                    } catch (...) {
                        train_targets.push_back(0.0);
                    }
                    break;
                }
            }
        }
        train_stream.close();

        cout << "Train: загружено " << train_features.size()
             << " примеров, " << train_features[0].size() << " признаков\n";

        int count0 = 0, count1 = 0;
        for (double t : train_targets) {
            if (t < 0.5) count0++;
            else count1++;
        }
        cout << "Класс 0: " << count0 << ", класс 1: " << count1 << "\n";
        cout << "Дисбаланс: " << fixed << setprecision(2)
             << (double)count1 / count0 << ":1 (class1/class0)\n";

        // Читаем test.csv
        ifstream test_stream(test_file);
        getline(test_stream, line); // пропускаем заголовок

        vector<vector<double>> test_features;

        while (getline(test_stream, line)) {
            if (line.empty()) continue;

            stringstream ss(line);
            string token;
            vector<string> row;

            while (getline(ss, token, ',')) {
                row.push_back(token);
            }

            vector<double> features = processCategoricalRow(row, headers);
            test_features.push_back(features);
        }
        test_stream.close();

        cout << "Test: загружено " << test_features.size()
             << " примеров, " << test_features[0].size() << " признаков\n";

        cout << "\n[2] Нормализация данных...\n";
        NormalizationParams norm_params = fitNormalizeData(train_features);
        normalizeDataWithParams(test_features, norm_params);
        cout << "Нормализация завершена\n";

        // Разделяем обучающие данные на train и validation (80/20)
        size_t total_samples = train_features.size();
        size_t val_size = total_samples / 5;
        size_t train_size = total_samples - val_size;

        vector<vector<double>> train_feat(train_features.begin(), train_features.begin() + train_size);
        vector<double> train_targ(train_targets.begin(), train_targets.begin() + train_size);
        vector<vector<double>> val_feat(train_features.begin() + train_size, train_features.end());
        vector<double> val_targ(train_targets.begin() + train_size, train_targets.end());

        // Преобразуем цели в vector<vector<double>> для Trainer
        vector<vector<double>> train_targets_vec, val_targets_vec;
        for (double t : train_targ) train_targets_vec.push_back({t});
        for (double t : val_targ) val_targets_vec.push_back({t});

        cout << "\n[3] Разделение данных:\n";
        cout << "  - Train samples: " << train_feat.size() << "\n";
        cout << "  - Validation samples: " << val_feat.size() << "\n";

        size_t n_features = train_feat[0].size();
        cout << "\n[4] Создание сети (архитектура: " << n_features << " -> 32 -> 16 -> 1)\n";
        NeuralNetwork net({(int)n_features, 32, 16, 1}, Activation::RELU, true, "log.txt");

        cout << "\n[5] Обучение...\n";
        TrainingConfig cfg;
        cfg.epochs = 300;
        cfg.learning_rate = 0.01;
        cfg.verbose = true;

        Trainer trainer(net, cfg);
        trainer.train(train_feat, train_targets_vec);

        // Оценка на валидационных данных
        Matrix<double> val_inputs_mat = vectorToMatrix(val_feat);
        Matrix<double> val_targets_mat = vectorToMatrix(val_targets_vec);
        double val_accuracy = trainer.evaluate(val_inputs_mat, val_targets_mat);
        cout << "\n[6] Точность на валидационных данных: " << fixed << setprecision(2)
             << val_accuracy << "%\n";

        // Оценка на обучающих данных
        Matrix<double> train_inputs_mat = vectorToMatrix(train_feat);
        Matrix<double> train_targets_mat = vectorToMatrix(train_targets_vec);
        double train_accuracy = trainer.evaluate(train_inputs_mat, train_targets_mat);
        cout << "Точность на обучающих данных: " << fixed << setprecision(2)
             << train_accuracy << "%\n";

        cout << "\n[7] Предсказание для тестовых данных...\n";

        ofstream submission("submission.csv");
        submission << "Id,Exited\n";

        for (size_t i = 0; i < test_features.size(); ++i) {
            vector<double> out = trainer.predict(test_features[i]);
            int pred = (out[0] > 0.5) ? 1 : 0;
            submission << i + 1 << "," << pred << "\n";
        }
        submission.close();

        cout << "  ✓ Предсказания сохранены в submission.csv\n";

        cout << "\n[8] Примеры предсказаний (первые 10):\n";
        for (size_t i = 0; i < min((size_t)10, test_features.size()); ++i) {
            vector<double> out = trainer.predict(test_features[i]);
            int pred = (out[0] > 0.5) ? 1 : 0;
            cout << "  Пример " << setw(4) << i + 1 << " -> P(1) = " << fixed << setprecision(4)
                 << out[0] << ", класс: " << pred << "\n";
        }

        net.saveModel("churn_model.txt");
        cout << "\n[9] Модель сохранена в churn_model.txt\n";

        cout << "\n=== Итоговая статистика ===\n";
        cout << "  - Лучшая точность на валидации: " << val_accuracy << "%\n";
        cout << "  - Файлы созданы: submission.csv, churn_model.txt, log.txt\n";

    } catch (const exception& e) {
        cerr << "\n❌ Ошибка: " << e.what() << "\n";
        return 1;
    }

    cout << "\n=== Программа успешно завершена ===\n";
    return 0;
}