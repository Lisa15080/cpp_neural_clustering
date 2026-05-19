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
#include <random>

#ifdef _WIN32
    #include <direct.h>
    #define getcwd _getcwd
#else
    #include <unistd.h>
#endif

using namespace std;

string getCurrentPath() {
    char buffer[1024];
#ifdef _WIN32
    if (_getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#endif
    return ".";
}

vector<vector<double>> matrixToVector(const Matrix<double>& mat) {
    vector<vector<double>> vec(mat.rows(), vector<double>(mat.cols()));
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

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
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            params.mean[j] += inputs[i][j];

    for (size_t j = 0; j < n_features; ++j)
        params.mean[j] /= n_samples;

    params.std.assign(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            params.std[j] += (inputs[i][j] - params.mean[j]) *
                             (inputs[i][j] - params.mean[j]);

    for (size_t j = 0; j < n_features; ++j) {
        params.std[j] = sqrt(params.std[j] / n_samples);
        if (params.std[j] < 1e-8) params.std[j] = 1.0;
    }

    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];

    return params;
}

void normalizeDataWithParams(vector<vector<double>>& inputs, const NormalizationParams& params) {
    for (size_t i = 0; i < inputs.size(); ++i)
        for (size_t j = 0; j < inputs[i].size(); ++j)
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];
}

vector<double> processCategoricalRow(const vector<string>& row, const vector<string>& headers) {
    vector<double> features;

    static const map<string, int> geography_map = {{"France", 0}, {"Spain", 1}, {"Germany", 2}};
    static const map<string, int> gender_map = {{"Female", 0}, {"Male", 1}};

    for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
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
            continue;
        }
        else if (header == "Exited") {
            continue;
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

double calcF1(Trainer& trainer,
              const vector<vector<double>>& X,
              const vector<double>& y,
              double threshold = 0.5) {
    int tp = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < X.size(); ++i) {
        vector<double> out = trainer.predict(X[i]);
        int pred = (out[0] > threshold) ? 1 : 0;
        int real = (y[i] > 0.5) ? 1 : 0;

        if (pred == 1 && real == 1) tp++;
        if (pred == 1 && real == 0) fp++;
        if (pred == 0 && real == 1) fn++;
    }

    double precision = (tp + fp == 0) ? 0.0 : (double)tp / (tp + fp);
    double recall = (tp + fn == 0) ? 0.0 : (double)tp / (tp + fn);

    if (precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

double runDataset(const string& train_file) {
    cout << "\n==============================\n";
    cout << "DATASET: " << train_file << "\n";
    cout << "==============================\n";

    CSVParser parser(',', true);
    vector<string> headers = parser.getHeaders(train_file);

    ifstream train_stream(train_file);
    if (!train_stream.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + train_file);
    }

    string line;
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

        vector<double> features = processCategoricalRow(row, headers);
        train_features.push_back(features);

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

    cout << "Загружено " << train_features.size()
         << " примеров, " << train_features[0].size() << " признаков\n";

    int count0 = 0, count1 = 0;
    for (double t : train_targets) {
        if (t < 0.5) count0++;
        else count1++;
    }

    cout << "Класс 0: " << count0 << ", класс 1: " << count1 << "\n";

    cout << "\n[1] Нормализация данных...\n";
    fitNormalizeData(train_features);

    vector<size_t> idx(train_features.size());
    iota(idx.begin(), idx.end(), 0);

    mt19937 gen(42);
    shuffle(idx.begin(), idx.end(), gen);

    vector<vector<double>> shuffled_features;
    vector<double> shuffled_targets;

    for (size_t id : idx) {
        shuffled_features.push_back(train_features[id]);
        shuffled_targets.push_back(train_targets[id]);
    }

    train_features = shuffled_features;
    train_targets = shuffled_targets;

    size_t total_samples = train_features.size();
    size_t val_size = total_samples / 5;
    size_t train_size = total_samples - val_size;

    vector<vector<double>> train_feat(train_features.begin(), train_features.begin() + train_size);
    vector<double> train_targ(train_targets.begin(), train_targets.begin() + train_size);

    vector<vector<double>> val_feat(train_features.begin() + train_size, train_features.end());
    vector<double> val_targ(train_targets.begin() + train_size, train_targets.end());

    vector<vector<double>> train_targets_vec;
    for (double t : train_targ) {
        train_targets_vec.push_back({t});
    }

    cout << "\n[2] Разделение данных:\n";
    cout << "Train samples: " << train_feat.size() << "\n";
    cout << "Validation samples: " << val_feat.size() << "\n";

    size_t n_features = train_feat[0].size();

    cout << "\n[3] Создание сети: "
         << n_features << " -> 32 -> 16 -> 1\n";

    NeuralNetwork net({(int)n_features, 32, 16, 1}, Activation::RELU, true, "log.txt");

    TrainingConfig cfg;
    cfg.epochs = 300;
    cfg.learning_rate = 0.01;
    cfg.verbose = true;

    cout << "\n[4] Обучение...\n";

    Trainer trainer(net, cfg);
    trainer.train(train_feat, train_targets_vec);

    double train_f1 = calcF1(trainer, train_feat, train_targ, 0.5);
    double val_f1 = calcF1(trainer, val_feat, val_targ, 0.5);

    cout << "\n[5] Результаты:\n";
    cout << "F1 на train:      " << fixed << setprecision(4) << train_f1 << "\n";
    cout << "F1 на validation: " << fixed << setprecision(4) << val_f1 << "\n";

    return val_f1;
}

int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    cout << "=== Нейронная сеть - проверка F1 на двух датасетах ===\n";
    cout << "Текущая папка: " << getCurrentPath() << "\n";

    try {
        vector<string> datasets = {
            "../datasets/dataset1.csv",
            "../datasets/dataset2.csv"
        };

        vector<double> scores;

        for (const string& file : datasets) {
            double f1 = runDataset(file);
            scores.push_back(f1);
        }

        if (scores.size() == 2) {
            double final_score = 0.5 * scores[0] + 0.5 * scores[1];

            cout << "\n==============================\n";
            cout << "ИТОГОВЫЙ SCORE\n";
            cout << "==============================\n";
            cout << "F1(d1): " << fixed << setprecision(4) << scores[0] << "\n";
            cout << "F1(d2): " << fixed << setprecision(4) << scores[1] << "\n";
            cout << "0.5 * F1(d1) + 0.5 * F1(d2) = "
                 << fixed << setprecision(4) << final_score << "\n";

            if (final_score >= 0.55) {
                cout << "Статус: проходит порог 0.55\n";
            } else {
                cout << "Статус: ниже порога 0.55\n";
            }
        }

    } catch (const exception& e) {
        cerr << "\nОшибка: " << e.what() << "\n";
        return 1;
    }

    cout << "\n=== Программа успешно завершена ===\n";
    return 0;
}