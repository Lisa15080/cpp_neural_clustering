#include "neuro_model/Neural_Net/neural_net.h"
#include "Trainer_class/trainer.h"
#include "class/Matrix/matrix.h"
#include "parser/pars.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#ifdef _WIN32
    #include <direct.h>
#endif

using namespace std;

// Вспомогательные функции

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

// Преобразовать Matrix<double> в vector<vector<double>> (нужно для Trainer)
vector<vector<double>> matrixToVector(const Matrix<double>& mat) {
    vector<vector<double>> vec(mat.rows(), vector<double>(mat.cols()));
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// Структура для хранения разбиения на train/test
struct DataSplit {
    Matrix<double> X_train, X_test;
    Matrix<double> y_train, y_test;
};

// Функция разбиения матриц на обучающую и тестовую выборки
DataSplit trainTestSplit(const Matrix<double>& X, const Matrix<double>& y,
                         double test_ratio = 0.2, bool shuffle = true) {
    size_t n = X.rows();
    vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);   // 0, 1, 2, ..., n-1

    if (shuffle) {
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

// Главная программа
int main() {
    cout << "=== Нейронная сеть + Trainer ===\n";
    cout << "Текущая папка: " << getCurrentPath() << "\n\n";

    // Загрузка данных из CSV
    cout << "[1] Загрузка данных из CSV...\n";
    CSVParser parser(',', true);   // разделитель запятая, первая строка – заголовок
    string filename = "Kegal_dataset/circles_dataset.csv";   // путь к реальному файлу

    try {
        // Загружаем данные для 2D классификации (формат: x, y, class)
        Datasetpars<double> csv_dataset = parser.loadClassification2D(filename);
        cout << "Загружено примеров: " << csv_dataset.inputs.rows()
             << ", признаков: " << csv_dataset.inputs.cols() << "\n";

        // Разделяем на обучающую и тестовую выборки (80% / 20%)
        DataSplit split = trainTestSplit(csv_dataset.inputs, csv_dataset.targets, 0.2, true);
        cout << "Обучающих примеров: " << split.X_train.rows()
             << ", тестовых: " << split.X_test.rows() << "\n";

        // Преобразуем матрицы в vector<vector<double>> для Trainer
        vector<vector<double>> train_inputs  = matrixToVector(split.X_train);
        vector<vector<double>> train_targets = matrixToVector(split.y_train);
        vector<vector<double>> test_inputs   = matrixToVector(split.X_test);
        vector<vector<double>> test_targets  = matrixToVector(split.y_test);

        // Создание сети
        cout << "\n[2] Создание сети (архитектура: 2 -> 16 -> 8 -> 1)\n";
        // Скрытые слои используют ReLU, выходной слой – сигмоида
        NeuralNetwork net({2, 16, 8, 1}, Activation::RELU, true, "log.txt");

        // 3. Обучение через Trainer 
        cout << "\n[3] Обучение...\n";
        TrainingConfig cfg;
        cfg.epochs = 500;           // количество эпох
        cfg.learning_rate = 0.05;   // скорость обучения
        cfg.verbose = true;         // печатать прогресс

        Trainer trainer(net, cfg);
        trainer.train(train_inputs, train_targets);

        // Оценка точности
        double accuracy = trainer.evaluate(test_inputs, test_targets);
        cout << "\n[4] Точность на тестовых данных: " << fixed << setprecision(2)
             << accuracy << "%\n";

        // Демонстрация предсказаний
        cout << "\n[5] Примеры предсказаний для нескольких точек:\n";
        vector<vector<double>> samples = {
            {0.0, 0.0},
            {0.5, 0.5},
            {-0.5, -0.5},
            {0.8, -0.2}
        };

        for (const auto& pt : samples) {
            vector<double> out = trainer.predict(pt);
            int pred = (out[0] > 0.5) ? 1 : 0;
            cout << "  Точка (" << setw(5) << pt[0] << ", " << setw(5) << pt[1]
                 << ") -> вероятность класса 1: " << fixed << setprecision(4) << out[0]
                 << ", предсказанный класс: " << pred << "\n";
        }

        // Сохраняем обученную модель
        net.saveModel("trained_model.txt");
        cout << "\nМодель сохранена в файл: trained_model.txt\n";

    } catch (const exception& e) {
        cerr << "Ошибка при работе с CSV: " << e.what() << "\n";
        cerr << "Проверьте, что файл " << filename << " существует и имеет правильный формат.\n";
        cerr << "Продолжаем с тестом XOR...\n";
    }

    // 6. Тест XOR (для проверки работоспособности)
    cout << "\n[6] Тест XOR (проверка базовой работы сети)\n";

    // Данные XOR
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

    // Преобразуем в векторы
    vector<vector<double>> xor_in_vec = matrixToVector(xor_inputs);
    vector<vector<double>> xor_tar_vec = matrixToVector(xor_targets);

    // Создаём небольшую сеть (скрытые слои – сигмоида)
    NeuralNetwork xor_net({2, 4, 1}, Activation::SIGMOID, false);

    // Настройки обучения для XOR
    TrainingConfig xor_cfg;
    xor_cfg.epochs = 2000;
    xor_cfg.learning_rate = 0.5;
    xor_cfg.verbose = true;

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