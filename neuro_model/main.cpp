#include "Neural_Net/neural_net.h"
#include "Trainer_class/trainer.h"
#include "DataSet/dataset.h"
#include "../class/Matrix/matrix.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

using namespace std;

string getCurrentPath() {
    char buffer[1024];
#ifdef _WIN32
    if (_getcwd(buffer, sizeof(buffer)) != nullptr) {
        return string(buffer);
    }
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) {
        return string(buffer);
    }
#endif
    return ".";
}

// Вспомогательная функция для преобразования vector<vector<double>> в Matrix<double>
Matrix<double> vectorToMatrix(const vector<vector<double>>& vec) {
    if (vec.empty()) return Matrix<double>();

    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Matrix<double> mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = vec[i][j];
        }
    }
    return mat;
}

// Вспомогательная функция для преобразования Matrix<double> в vector<vector<double>>
vector<vector<double>> matrixToVector(const Matrix<double>& mat) {
    vector<vector<double>> vec(mat.rows(), vector<double>(mat.cols()));

    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            vec[i][j] = mat(i, j);
        }
    }
    return vec;
}

int main() {
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif

    cout << "Тест нейросети\n";
    cout << "========================\n";

    // Генерация данных
    DatasetGenerator generator;
    Dataset data = generator.generate_gaussian(100, 0.5, 2.0);

    cout << "\nСгенерировано гауссовых кластеров: " << data.inputs.size() << " точек\n";

    // Данные для XOR в виде матриц
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

    // СЕТЬ 1: XOR
    cout << "\n[1] Создаём сеть для XOR: 2→3→1\n";
    NeuralNetwork net({2, 3, 1}, true, "network_log.txt");

    cout << "\nРезультаты ДО обучения:\n";
    for (size_t i = 0; i < xor_inputs.rows(); i++) {
        // Преобразуем строку матрицы в вектор для forward
        vector<double> input_row = {xor_inputs(i, 0), xor_inputs(i, 1)};
        auto res = net.forward(input_row);
        cout << "  [" << xor_inputs(i, 0) << "," << xor_inputs(i, 1)
             << "] → " << fixed << setprecision(4) << res[0] << "\n";
    }

    cout << "\nОбучаем сеть на XOR (1000 эпох):\n";
    TrainingConfig cfg_xor;
    cfg_xor.epochs = 1000;
    cfg_xor.learning_rate = 0.5;
    cfg_xor.verbose = true;

    // Преобразуем матрицы обратно в векторы для Trainer (если он еще не поддерживает Matrix)
    vector<vector<double>> xor_inputs_vec = matrixToVector(xor_inputs);
    vector<vector<double>> xor_targets_vec = matrixToVector(xor_targets);

    Trainer trainer_xor(net, cfg_xor);
    trainer_xor.train(xor_inputs_vec, xor_targets_vec);

    cout << "\nРезультаты ПОСЛЕ обучения:\n";
    for (size_t i = 0; i < xor_inputs.rows(); i++) {
        vector<double> input_row = {xor_inputs(i, 0), xor_inputs(i, 1)};
        auto res = net.forward(input_row);
        cout << "  [" << xor_inputs(i, 0) << "," << xor_inputs(i, 1)
             << "] → " << fixed << setprecision(4) << res[0]
             << " (ожидаем: " << xor_targets(i, 0) << ")\n";
    }

    // СЕТЬ 2: ГАУССОВЫ КЛАСТЕРЫ
    cout << "\n\n[2] Создаём сеть для гауссовых кластеров: 2→5→1\n";
    NeuralNetwork net2({2, 5, 1}, true, "network_log2.txt");

    // Преобразуем данные гауссовых кластеров в матрицы
    Matrix<double> gauss_inputs = vectorToMatrix(data.inputs);
    Matrix<double> gauss_targets = vectorToMatrix(data.targets);

    cout << "Обучаем на гауссовых кластерах (500 эпох):\n";
    TrainingConfig cfg_gauss;
    cfg_gauss.epochs = 500;
    cfg_gauss.learning_rate = 0.1;
    cfg_gauss.verbose = true;

    Trainer trainer_gauss(net2, cfg_gauss);
    trainer_gauss.train(data.inputs, data.targets);  // Используем исходные векторы

    // Проверка точности
    double accuracy = trainer_gauss.evaluate(data);
    cout << "\nТочность на гауссовых кластерах: " << fixed << setprecision(2) << accuracy << "%\n";

    // Сохранение/загрузка
    cout << "\n[3] Сохраняем модель...\n";
    if (net.saveModel("model.txt")) {
        cout << "Модель сохранена: " << getCurrentPath() << "/model.txt\n";
    }

    cout << "\nЗагружаем и проверяем:\n";
    NeuralNetwork net_loaded({2, 3, 1}, false);
    if (net_loaded.loadModel("model.txt")) {
        cout << "Модель загружена успешно!\n";
        cout << "Проверка на XOR:\n";
        for (size_t i = 0; i < xor_inputs.rows(); i++) {
            vector<double> input_row = {xor_inputs(i, 0), xor_inputs(i, 1)};
            auto res = net_loaded.forward(input_row);
            cout << "  [" << xor_inputs(i, 0) << "," << xor_inputs(i, 1)
                 << "] → " << fixed << setprecision(4) << res[0] << "\n";
        }
    }

    cout << "\n========================\n";
    cout << "Работа завершена!\n";

    return 0;
}