#include "Neural_Net/neural_net.h"
#include "../Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"
#include "../parser/pars.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <cmath>

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

// Функция для очистки строки от кавычек и пробелов
string cleanString(const string& str) {
    string result = str;
    result.erase(0, result.find_first_not_of(" \t\r\n"));
    result.erase(result.find_last_not_of(" \t\r\n") + 1);
    if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
        result = result.substr(1, result.size() - 2);
    }
    return result;
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

// Структура для хранения данных после предобработки
struct ProcessedData {
    vector<vector<double>> features; // признаки
    vector<vector<double>> targets;  // целевая переменная
};

// Загрузка train.csv
ProcessedData loadTrainData(const string& filename, char delimiter = ',') {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<vector<double>> data_rows;
    vector<double> target_values;
    vector<string> headers;
    string line;

    getline(file, line);
    stringstream header_ss(line);
    string header;
    while (getline(header_ss, header, delimiter)) {
        headers.push_back(cleanString(header));
    }

    vector<int> feature_indices;
    int target_index = -1;

    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == "Exited") {
            target_index = i;
        } else if (headers[i] != "RowNumber" &&
                   headers[i] != "CustomerId" &&
                   headers[i] != "Surname") {
            feature_indices.push_back(i);
        }
    }

    map<string, int> geography_map = {{"France", 0}, {"Spain", 1}, {"Germany", 2}};
    map<string, int> gender_map = {{"Female", 0}, {"Male", 1}};

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss_line(line);
        string token;
        vector<string> row_str;

        while (getline(ss_line, token, delimiter)) {
            row_str.push_back(cleanString(token));
        }

        vector<double> row_features;

        for (int idx : feature_indices) {
            string value = row_str[idx];

            if (headers[idx] == "Geography") {
                row_features.push_back(geography_map[value]);
            }
            else if (headers[idx] == "Gender") {
                row_features.push_back(gender_map[value]);
            }
            else {
                try {
                    row_features.push_back(stod(value));
                } catch (...) {
                    row_features.push_back(0.0);
                }
            }
        }

        data_rows.push_back(row_features);

        if (target_index != -1) {
            try {
                target_values.push_back(stod(row_str[target_index]));
            } catch (...) {
                target_values.push_back(0.0);
            }
        }
    }

    file.close();

    ProcessedData result;
    result.features = data_rows;

    // Convert target_values to vector<vector<double>>
    for (double val : target_values) {
        result.targets.push_back({val});
    }

    return result;
}

// Загрузка test.csv (без целевой переменной)
vector<vector<double>> loadTestData(const string& filename, char delimiter = ',') {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<vector<double>> data_rows;
    vector<string> headers;
    string line;

    getline(file, line);
    stringstream header_ss(line);
    string header;
    while (getline(header_ss, header, delimiter)) {
        headers.push_back(cleanString(header));
    }

    vector<int> feature_indices;
    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] != "RowNumber" &&
            headers[i] != "CustomerId" &&
            headers[i] != "Surname") {
            feature_indices.push_back(i);
        }
    }

    map<string, int> geography_map = {{"France", 0}, {"Spain", 1}, {"Germany", 2}};
    map<string, int> gender_map = {{"Female", 0}, {"Male", 1}};

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss_line(line);
        string token;
        vector<string> row_str;

        while (getline(ss_line, token, delimiter)) {
            row_str.push_back(cleanString(token));
        }

        vector<double> row_features;

        for (int idx : feature_indices) {
            string value = row_str[idx];

            if (headers[idx] == "Geography") {
                row_features.push_back(geography_map[value]);
            }
            else if (headers[idx] == "Gender") {
                row_features.push_back(gender_map[value]);
            }
            else {
                try {
                    row_features.push_back(stod(value));
                } catch (...) {
                    row_features.push_back(0.0);
                }
            }
        }

        data_rows.push_back(row_features);
    }

    file.close();
    return data_rows;
}

// Функция для ручной оценки точности
double computeAccuracy(Trainer& trainer,
                       const vector<vector<double>>& inputs,
                       const vector<vector<double>>& targets) {
    if (inputs.size() != targets.size()) return 0.0;

    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<double> out = trainer.predict(inputs[i]);
        int predicted = (out[0] > 0.5) ? 1 : 0;
        int actual = static_cast<int>(targets[i][0] + 0.5);
        if (predicted == actual) correct++;
    }
    return 100.0 * correct / inputs.size();
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

        ProcessedData train_data = loadTrainData(train_file, ',');

        cout << "Train: загружено " << train_data.features.size()
             << " примеров, " << train_data.features[0].size() << " признаков\n";

        int count0 = 0, count1 = 0;
        for (const auto& t : train_data.targets) {
            if (t[0] < 0.5) count0++;
            else count1++;
        }
        cout << "Класс 0: " << count0 << ", класс 1: " << count1 << "\n";

        vector<vector<double>> test_features = loadTestData(test_file, ',');
        cout << "Test: загружено " << test_features.size()
             << " примеров, " << test_features[0].size() << " признаков\n";

        cout << "\n[2] Нормализация данных...\n";
        normalizeData(train_data.features);
        normalizeData(test_features);
        cout << "Нормализация завершена\n";

        size_t n_features = train_data.features[0].size();
        cout << "\n[3] Создание сети (архитектура: " << n_features << " -> 64 -> 32 -> 16 -> 1)\n";
        NeuralNetwork net({(int)n_features, 32, 16, 1}, Activation::RELU, true, "log.txt");

        cout << "\n[4] Обучение...\n";
        TrainingConfig cfg;
        cfg.epochs = 300;
        cfg.learning_rate = 0.01;
        cfg.verbose = true;

        Trainer trainer(net, cfg);
        trainer.train(train_data.features, train_data.targets);

        double train_accuracy = computeAccuracy(trainer, train_data.features, train_data.targets);
        cout << "\n[5] Точность на обучающих данных: " << fixed << setprecision(2)
             << train_accuracy << "%\n";

        cout << "\n[6] Предсказание для тестовых данных...\n";

        ofstream submission("submission.csv");
        submission << "Id,Exited\n";

        for (size_t i = 0; i < test_features.size(); ++i) {
            vector<double> out = trainer.predict(test_features[i]);
            int pred = (out[0] > 0.5) ? 1 : 0;
            submission << i + 1 << "," << pred << "\n";
        }
        submission.close();

        cout << "Предсказания сохранены в submission.csv\n";

        cout << "\n[7] Примеры предсказаний (первые 10):\n";
        for (size_t i = 0; i < min((size_t)10, test_features.size()); ++i) {
            vector<double> out = trainer.predict(test_features[i]);
            int pred = (out[0] > 0.5) ? 1 : 0;
            cout << "  Пример " << i + 1 << " -> P(1) = " << fixed << setprecision(4) << out[0]
                 << ", класс: " << pred << "\n";
        }

        net.saveModel("churn_model.txt");
        cout << "\nМодель сохранена в churn_model.txt\n";

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    cout << "\n=== Программа завершена ===\n";
    return 0;
}