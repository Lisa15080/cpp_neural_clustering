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
    // Удаляем пробелы в начале и конце
    result.erase(0, result.find_first_not_of(" \t\r\n"));
    result.erase(result.find_last_not_of(" \t\r\n") + 1);
    // Удаляем кавычки в начале и конце если они есть
    if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
        result = result.substr(1, result.size() - 2);
    }
    return result;
}

// Структура для хранения разбиения на train/test
struct DataSplit {
    Matrix<double> X_train, X_test;
    Matrix<double> y_train, y_test;
};

// Функция разбиения матриц на обучающую и тестовую выборки
DataSplit trainTestSplit(const Matrix<double>& X, const Matrix<double>& y,
                         double test_ratio, bool do_shuffle) {
    size_t n = X.rows();
    vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);

    if (do_shuffle) {
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);
    }

    size_t test_size = static_cast<size_t>(n * test_ratio);
    size_t train_size = n - test_size;

    Matrix<double> X_train(train_size, X.cols());
    Matrix<double> X_test(test_size, X.cols());
    Matrix<double> y_train(train_size, y.cols());
    Matrix<double> y_test(test_size, y.cols());

    for (size_t i = 0; i < train_size; ++i) {
        size_t idx = indices[i];
        for (size_t j = 0; j < X.cols(); ++j) X_train(i, j) = X(idx, j);
        for (size_t j = 0; j < y.cols(); ++j) y_train(i, j) = y(idx, j);
    }
    for (size_t i = 0; i < test_size; ++i) {
        size_t idx = indices[train_size + i];
        for (size_t j = 0; j < X.cols(); ++j) X_test(i, j) = X(idx, j);
        for (size_t j = 0; j < y.cols(); ++j) y_test(i, j) = y(idx, j);
    }
    return {X_train, X_test, y_train, y_test};
}

// Загрузка CSV с автоматической очисткой кавычек
struct CleanDataset {
    Matrix<double> inputs;
    Matrix<double> targets;
};

CleanDataset loadCSVWithCleaning(const string& filename, char delimiter = ',') {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + filename);
    }

    vector<vector<double>> data_rows;
    vector<string> lines;
    string line;

    // Пропускаем заголовок
    getline(file, line);

    // Читаем все строки
    while (getline(file, line)) {
        if (line.empty()) continue;
        lines.push_back(line);
    }
    file.close();

    if (lines.empty()) {
        throw runtime_error("Файл пуст или содержит только заголовок: " + filename);
    }

    // Определяем количество столбцов по первой строке
    stringstream ss(lines[0]);
    string token;
    size_t cols = 0;
    while (getline(ss, token, delimiter)) cols++;

    // Парсим данные
    for (const auto& l : lines) {
        stringstream ss_line(l);
        vector<double> row;
        size_t col_idx = 0;

        while (getline(ss_line, token, delimiter)) {
            string cleaned = cleanString(token);
            if (cleaned.empty()) {
                row.push_back(0.0);
            } else {
                try {
                    row.push_back(stod(cleaned));
                } catch (...) {
                    row.push_back(0.0);
                }
            }
            col_idx++;
        }

        if (!row.empty()) {
            data_rows.push_back(row);
        }
    }

    if (data_rows.empty()) {
        throw runtime_error("Не удалось распарсить данные из файла: " + filename);
    }

    size_t rows = data_rows.size();
    size_t input_cols = cols - 1; // последний столбец - метка класса

    CleanDataset result;
    result.inputs = Matrix<double>(rows, input_cols);
    result.targets = Matrix<double>(rows, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < input_cols; ++j) {
            result.inputs(i, j) = data_rows[i][j];
        }
        result.targets(i, 0) = data_rows[i][input_cols];
    }

    return result;
}

// Главная программа
int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");
#endif

    cout << "=== Нейронная сеть + Trainer (бинарная классификация) ===\n";
    cout << "Текущая папка: " << getCurrentPath() << "\n\n";

    // Загрузка данных из CSV
    cout << "[1] Загрузка данных из CSV...\n";
    string filename = "../Kegal_detaset/circles_detaset.csv";

    try {
        // Загружаем данные с автоматической очисткой от кавычек
        CleanDataset clean_data = loadCSVWithCleaning(filename, ',');

        cout << "Загружено примеров: " << clean_data.inputs.rows()
             << ", признаков: " << clean_data.inputs.cols() << "\n";

        // Определяем уникальные классы в датасете
        set<int> unique_classes;
        for (size_t i = 0; i < clean_data.targets.rows(); ++i) {
            unique_classes.insert(static_cast<int>(clean_data.targets(i, 0) + 0.5));
        }
        cout << "Найдено классов: " << unique_classes.size() << " (";
        for (auto it = unique_classes.begin(); it != unique_classes.end(); ++it) {
            if (it != unique_classes.begin()) cout << ", ";
            cout << *it;
        }
        cout << ")\n";

        // Если классов больше двух, преобразуем в бинарные
        // Берём первый класс как 0, остальные как 1
        Matrix<double> binary_targets(clean_data.targets.rows(), 1);
        int class0 = *unique_classes.begin();
        int count0 = 0, count1 = 0;

        for (size_t i = 0; i < clean_data.targets.rows(); ++i) {
            int label = static_cast<int>(clean_data.targets(i, 0) + 0.5);
            if (label == class0) {
                binary_targets(i, 0) = 0.0;
                count0++;
            } else {
                binary_targets(i, 0) = 1.0;
                count1++;
            }
        }

        cout << "Преобразовано в бинарные метки: класс 0 = " << count0
             << ", класс 1 = " << count1 << "\n";

        // Разделяем на обучающую и тестовую выборки
        DataSplit split = trainTestSplit(clean_data.inputs, binary_targets, 0.2, true);
        cout << "Обучающих примеров: " << split.X_train.rows()
             << ", тестовых: " << split.X_test.rows() << "\n";

        // Преобразуем матрицы в vector<vector<double>> для Trainer
        vector<vector<double>> train_inputs  = matrixToVector(split.X_train);
        vector<vector<double>> train_targets = matrixToVector(split.y_train);
        vector<vector<double>> test_inputs   = matrixToVector(split.X_test);
        vector<vector<double>> test_targets  = matrixToVector(split.y_test);

        // Создание сети для бинарной классификации
        cout << "\n[2] Создание сети (архитектура: 2 -> 16 -> 8 -> 1)\n";
        NeuralNetwork net({2, 16, 8, 1}, Activation::RELU, true, "log.txt");

        // Обучение через Trainer
        cout << "\n[3] Обучение...\n";
        TrainingConfig cfg;
        cfg.epochs = 300;
        cfg.learning_rate = 0.05;
        cfg.verbose = true;

        Trainer trainer(net, cfg);
        trainer.train(train_inputs, train_targets);

        // Оценка точности
        double accuracy = trainer.evaluate(test_inputs, test_targets);
        cout << "\n[4] Точность на тестовых данных: " << fixed << setprecision(2)
             << accuracy << "%\n";

        // Демонстрация предсказаний
        cout << "\n[5] Примеры предсказаний:\n";

        // Находим границы данных для красивого вывода
        double min_x = clean_data.inputs(0, 0), max_x = clean_data.inputs(0, 0);
        double min_y = clean_data.inputs(0, 1), max_y = clean_data.inputs(0, 1);
        for (size_t i = 0; i < clean_data.inputs.rows(); ++i) {
            min_x = min(min_x, clean_data.inputs(i, 0));
            max_x = max(max_x, clean_data.inputs(i, 0));
            min_y = min(min_y, clean_data.inputs(i, 1));
            max_y = max(max_y, clean_data.inputs(i, 1));
        }

        vector<vector<double>> samples = {
            {min_x, min_y},
            {(min_x + max_x) / 2, (min_y + max_y) / 2},
            {max_x, max_y},
            {min_x, max_y},
            {max_x, min_y}
        };

        for (const auto& pt : samples) {
            vector<double> out = trainer.predict(pt);
            int pred = (out[0] > 0.5) ? 1 : 0;
            cout << "  Точка (" << setw(8) << fixed << setprecision(2) << pt[0]
                 << ", " << setw(8) << pt[1]
                 << ") -> P(1) = " << setprecision(4) << out[0]
                 << ", класс: " << pred << "\n";
        }

        // Сохраняем обученную модель
        net.saveModel("trained_model.txt");
        cout << "\nМодель сохранена в файл: trained_model.txt\n";

    } catch (const exception& e) {
        cerr << "Ошибка при работе с CSV: " << e.what() << "\n";
        cerr << "Проверьте, что файл " << filename << " существует и имеет правильный формат.\n";
        cerr << "Продолжаем с тестом XOR...\n";
    }

    // Тест XOR для проверки работоспособности
    cout << "\n[6] Тест XOR (проверка базовой работы сети)\n";

    Matrix<double> xor_inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    Matrix<double> xor_targets = {
        {0},
        {1},
        {1},
        {0}
    };

    vector<vector<double>> xor_in_vec = matrixToVector(xor_inputs);
    vector<vector<double>> xor_tar_vec = matrixToVector(xor_targets);

    NeuralNetwork xor_net({2, 4, 1}, Activation::SIGMOID, false);

    TrainingConfig xor_cfg;
    xor_cfg.epochs = 2000;
    xor_cfg.learning_rate = 0.5;
    xor_cfg.verbose = false;  // меньше вывода для XOR

    Trainer xor_trainer(xor_net, xor_cfg);
    xor_trainer.train(xor_in_vec, xor_tar_vec);

    cout << "\nРезультаты XOR после обучения:\n";
    for (size_t i = 0; i < xor_inputs.rows(); ++i) {
        vector<double> in = {xor_inputs(i, 0), xor_inputs(i, 1)};
        vector<double> out = xor_trainer.predict(in);
        int pred = (out[0] > 0.5) ? 1 : 0;
        cout << "  [" << xor_inputs(i, 0) << ", " << xor_inputs(i, 1)
             << "] -> " << fixed << setprecision(4) << out[0]
             << " (класс " << pred << ", ожидалось " << static_cast<int>(xor_targets(i, 0)) << ")\n";
    }

    cout << "\n=== Программа завершена ===\n";
    return 0;
}
