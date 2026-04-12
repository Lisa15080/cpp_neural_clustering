#include "neuro_model/Neural_Net/neural_net.h"
#include "Trainer_class/trainer.h"
#include "DataSet/dataset.h"
#include "class/Matrix/matrix.h"
#include "parser/pars.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <vector>

// Windows.h нужно включать ПОСЛЕ всех стандартных библиотек
#ifdef _WIN32
    #include <direct.h>
    // Временно закомментируйте windows.h
    // #include <windows.h>
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

// Вспомогательные функции
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
        //SetConsoleOutputCP(CP_UTF8);
        //SetConsoleCP(CP_UTF8);
    #endif

    cout << "Тест нейросети\n";
    cout << "========================\n";

    // ==================== 1. ЗАГРУЗКА ДАННЫХ ИЗ CSV ====================
    cout << "\n[1] Загрузка данных из CSV файла...\n";

    CSVParser parser(',', true);

    // Путь к файлу с данными (создайте такой файл или измените путь)
    string filename = "data/points.csv";

    try {
        // Используем Datasetpars (новое название)
        Datasetpars<double> csv_dataset = parser.loadClassification2D(filename);

        cout << "Данные успешно загружены из файла: " << filename << endl;
        cout << "Всего samples: " << csv_dataset.inputs.rows() << endl;
        cout << "Признаков: " << csv_dataset.inputs.cols() << endl;
        cout << "Целей: " << csv_dataset.targets.cols() << endl;

        // Преобразуем Matrix<double> в vector<vector<double>> для Trainer
        vector<vector<double>> train_inputs = matrixToVector(csv_dataset.inputs);
        vector<vector<double>> train_targets = matrixToVector(csv_dataset.targets);

        // ==================== 2. СОЗДАЕМ СЕТЬ ====================
        cout << "\n[2] Создаём сеть для данных из CSV: 2→5→1\n";
        NeuralNetwork net({2, 5, 1}, true, "network_log.txt");

        cout << "\nОбучаем на загруженных данных (500 эпох):\n";
        TrainingConfig cfg;
        cfg.epochs = 500;
        cfg.learning_rate = 0.1;
        cfg.verbose = true;

        Trainer trainer(net, cfg);
        trainer.train(train_inputs, train_targets);

        // ==================== 3. ОЦЕНКА ТОЧНОСТИ ====================
        cout << "\n[3] Оценка точности...\n";

        auto split = parser.splitTrainTest(csv_dataset.inputs, csv_dataset.targets, 0.2, true);

        vector<vector<double>> X_test_vec = matrixToVector(split.X_test);
        vector<vector<double>> y_test_vec = matrixToVector(split.y_test);

        double accuracy = trainer.evaluate(X_test_vec, y_test_vec);
        cout << "Точность на тестовых данных: " << fixed << setprecision(2) << accuracy << "%\n";

        // ==================== 4. ПРЕДСКАЗАНИЕ ====================
        cout << "\n[4] Предсказание для новых точек:\n";
        vector<vector<double>> new_points = {
            {0.5, 0.5},
            {-0.5, 0.5},
            {0.5, -0.5},
            {0.0, 0.0}
        };

        for (const auto& point : new_points) {
            auto prediction = net.forward(point);
            cout << "  Точка (" << point[0] << ", " << point[1]
                 << ") → " << fixed << setprecision(4) << prediction[0];
            if (prediction[0] > 0.5) {
                cout << " (класс 1)";
            } else {
                cout << " (класс 0)";
            }
            cout << endl;
        }

    } catch (const exception& e) {
        cerr << "Ошибка: " << e.what() << endl;
        cout << "\nПродолжаем с тестом XOR...\n";
    }

    // ==================== 5. XOR ДЛЯ СРАВНЕНИЯ ====================
    cout << "\n[5] Тест XOR (для проверки работоспособности сети)\n";

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

    NeuralNetwork xor_net({2, 3, 1}, true, "xor_log.txt");

    cout << "\nРезультаты ДО обучения XOR:\n";
    for (size_t i = 0; i < xor_inputs.rows(); i++) {
        vector<double> input_row = {xor_inputs(i, 0), xor_inputs(i, 1)};
        auto res = xor_net.forward(input_row);
        cout << "  [" << xor_inputs(i, 0) << "," << xor_inputs(i, 1)
             << "] → " << fixed << setprecision(4) << res[0] << "\n";
    }

    cout << "\nОбучаем сеть на XOR (1000 эпох):\n";
    TrainingConfig cfg_xor;
    cfg_xor.epochs = 1000;
    cfg_xor.learning_rate = 0.5;
    cfg_xor.verbose = true;

    vector<vector<double>> xor_inputs_vec = matrixToVector(xor_inputs);
    vector<vector<double>> xor_targets_vec = matrixToVector(xor_targets);

    Trainer xor_trainer(xor_net, cfg_xor);
    xor_trainer.train(xor_inputs_vec, xor_targets_vec);

    cout << "\nРезультаты ПОСЛЕ обучения XOR:\n";
    for (size_t i = 0; i < xor_inputs.rows(); i++) {
        vector<double> input_row = {xor_inputs(i, 0), xor_inputs(i, 1)};
        auto res = xor_net.forward(input_row);
        cout << "  [" << xor_inputs(i, 0) << "," << xor_inputs(i, 1)
             << "] → " << fixed << setprecision(4) << res[0]
             << " (ожидаем: " << xor_targets(i, 0) << ")\n";
    }

    // Сохранение модели
    cout << "\n[6] Сохраняем модель...\n";
    if (xor_net.saveModel("model.txt")) {
        cout << "Модель сохранена: " << getCurrentPath() << "/model.txt\n";
    }

    cout << "\n========================\n";
    cout << "Работа завершена!\n";

    return 0;
}