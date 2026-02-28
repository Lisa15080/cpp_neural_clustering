// подключаем заголовочный файл нашего класса
#include "neural_net.h"
// подключаем нужные библиотеки
#include <cmath> // для математических функций
#include <cstdlib> // для случайных чисел
#include <ctime> // чтобы случайные числа были разными

using namespace std;

// вспомогательная функция для логирования(записывает сообщение в консоль и в файл)
void NeuralNetwork::log(const string& message, bool toConsole) {
    // Если нужно вывести в консоль
    if (toConsole) {
        cout << message;
    }

    // Если логирование включено и файл открыт
    if (loggingEnabled && logFile.is_open()) {
        logFile << message; // пишем в файл
    }
}

// конструктор(само создание)
NeuralNetwork::NeuralNetwork(const vector<int>& sizes, bool enableLogging, const string& filename) {
    // делаем случайные числа каждый раз разными
    srand(time(nullptr));

    // настраиваем логирование
    loggingEnabled = enableLogging;
    logFilename = filename;

    // если включено логирование в файл - открываем файл и пишем заголовок
    if (loggingEnabled) {
        logFile.open(logFilename);
        logFile << "Лог нейросети" << endl;
        logFile << "Архитектура: ";
        for (int size : sizes) logFile << size << " ";
        logFile << endl << endl;
    }

    // формируем сообщение об архитектуре
    string archMsg = "Создаем нейросеть с архитектурой: ";
    for (int size : sizes) archMsg += to_string(size) + " ";
    log(archMsg + "\n");

    // создаем слои (количество слоев = sizes.size() - 1)
    for (size_t i = 0; i < sizes.size() - 1; i++) {
        Layer layer;  // создаем пустой слой

        int inputSize = sizes[i]; // сколько чисел приходит на вход этого слоя
        int outputSize = sizes[i + 1]; // сколько нейронов в этом слое

        string layerMsg = "Слой " + to_string(i) + ": " +
            to_string(inputSize) + " -> " +
            to_string(outputSize) + " нейронов\n";
        log(layerMsg);

        // выделяем память под веса: outputSize строк, inputSize столбцов
        layer.weights.resize(outputSize, vector<double>(inputSize));

        // выделяем память под смещения: у каждого нейрона одно смещение
        layer.biases.resize(outputSize, vector<double>(1));

        // заполняем веса и смещения случайными числами от -1 до 1
        for (int j = 0; j < outputSize; j++) { // для каждого нейрона
            for (int k = 0; k < inputSize; k++) { // для каждого входа
                layer.weights[j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            // смещение тоже случайное от -1 до 1
            layer.biases[j][0] = ((double)rand() / RAND_MAX) * 2 - 1;
        }

        // добавляем готовый слой в сеть
        layers.push_back(layer);
    }

    log("Нейросеть создана\n\n");
}

// функции активации

// сигмоида(превращает любое число в число от 0 до 1)
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// производная сигмоиды(нужна для обучения)
double NeuralNetwork::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// ReLU(если число положительное - оставляем, если отрицательное - делаем 0)
double NeuralNetwork::relu(double x) {
    if (x > 0) return x;
    return 0;
}

// производная ReLU(нужна для обучения)
double NeuralNetwork::reluDerivative(double x) {
    if (x > 0) return 1;
    return 0;
}

// проход forward(как сеть думает)
vector<double> NeuralNetwork::forward(const vector<double>& input) {
    // выводим входные данные
    string msg = "Прямой проход: входные данные = ";
    for (double val : input) msg += to_string(val) + " ";
    log(msg + "\n");

    // превращаем входной вектор в матрицу-столбец
    vector<vector<double>> current;
    current.resize(input.size(), vector<double>(1));
    for (size_t i = 0; i < input.size(); i++) {
        current[i][0] = input[i];
    }

    log("Преобразовали вход в матрицу:\n");
    for (size_t i = 0; i < current.size(); i++) {
        log("  [" + to_string(current[i][0]) + "]\n");
    }

    // проходим по всем слоям по очереди
    for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
        const Layer& layer = layers[layerIdx];

        // выводим информацию о слое
        string layerHeader = "Слой " + to_string(layerIdx) + ":\n";
        layerHeader += "  Веса (" + to_string(layer.weights.size()) + "x" +
            to_string(layer.weights[0].size()) + "):\n";
        log(layerHeader);

        // здесь будет результат после этого слоя
        vector<vector<double>> next;
        next.resize(layer.weights.size(), vector<double>(1, 0.0));

        // для каждого нейрона в слое
        for (size_t i = 0; i < layer.weights.size(); i++) {
            string neuronMsg = "    Нейрон " + to_string(i) + ": ";

            // считаем сумму входов, умноженных на веса
            for (size_t j = 0; j < layer.weights[i].size(); j++) {
                // добавляем вес*вход
                neuronMsg += to_string(layer.weights[i][j]) + "*" +
                    to_string(current[j][0]);
                if (j < layer.weights[i].size() - 1) neuronMsg += " + ";

                // считаем: вес * вход и добавляем к сумме
                next[i][0] += layer.weights[i][j] * current[j][0];
            }

            // добавляем смещение
            neuronMsg += " + " + to_string(layer.biases[i][0]) + " = ";
            next[i][0] += layer.biases[i][0];
            neuronMsg += to_string(next[i][0]);

            // применяем функцию активации (сигмоиду)
            neuronMsg += " -> сигмоида -> ";
            next[i][0] = sigmoid(next[i][0]);
            neuronMsg += to_string(next[i][0]) + "\n";

            // выводим всё что насчитали
            log(neuronMsg);
        }

        // результат этого слоя становится входом для следующего
        current = next;
    }

    // превращаем обратно из матрицы в обычный вектор
    vector<double> result;
    for (const auto& row : current) {
        result.push_back(row[0]);
    }

    // выводим результат
    string resultMsg = "Результат: ";
    for (double val : result) resultMsg += to_string(val) + " ";
    log(resultMsg + "\n------------------------\n");

    return result;
}

// печать слоев(для проверки)
void NeuralNetwork::printLayers() {
    string msg = "Структура нейросети:\n";
    log(msg);

    // проходим по всем слоям
    for (size_t l = 0; l < layers.size(); l++) {
        string layerMsg = "Слой " + to_string(l) + ":\n";
        log(layerMsg);

        // печатаем веса
        log("  Веса:\n");
        for (size_t i = 0; i < layers[l].weights.size(); i++) {
            string rowMsg = "    ";
            for (double val : layers[l].weights[i]) {
                rowMsg += to_string(val) + " ";
            }
            log(rowMsg + "\n");
        }

        // печатаем смещения
        log("  Смещения:\n    ");
        for (const auto& b : layers[l].biases) {
            log(to_string(b[0]) + " ");
        }
        log("\n");
    }
}

// сохранение модели
bool NeuralNetwork::saveModel(const string& filename) {
    // открываем файл для записи
    ofstream file(filename);

    // проверяем, открылся ли файл
    if (!file.is_open()) {
        log("Ошибка: Нельзя открыть файл для сохранения: " + filename + "\n");
        return false;
    }

    log("Сохраняем модель в файл: " + filename + "\n");

    // сохраняем количество слоев
    file << layers.size() << endl;

    // для каждого слоя
    for (const auto& layer : layers) {
        // сохраняем размеры
        file << layer.weights.size() << " " << layer.weights[0].size() << endl;

        // сохраняем все веса
        for (const auto& row : layer.weights) {
            for (double val : row) {
                file << val << " ";
            }
            file << endl;
        }

        // сохраняем все смещения
        for (const auto& b : layer.biases) {
            file << b[0] << " ";
        }
        file << endl;
    }

    // закрываем файл
    file.close();
    log("Модель сохранена\n");
    return true;
}

// загрузка модели из файла
bool NeuralNetwork::loadModel(const string& filename) {
    // открываем файл для чтения
    ifstream file(filename);

    // проверяем, открылся ли файл
    if (!file.is_open()) {
        log("Ошибка: Не могу открыть файл для загрузки: " + filename + "\n");
        return false;
    }

    log("Загружаем модель из файла: " + filename + "\n");

    // читаем количество слоев
    int numLayers;
    file >> numLayers;

    // очищаем текущие слои
    layers.clear();

    // загружаем каждый слой
    for (int l = 0; l < numLayers; l++) {
        Layer layer;

        // читаем размеры слоя
        int rows, cols;
        file >> rows >> cols;

        // выделяем память под веса и смещения
        layer.weights.resize(rows, vector<double>(cols));
        layer.biases.resize(rows, vector<double>(1));

        // читаем веса
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file >> layer.weights[i][j];
            }
        }

        // читаем смещения
        for (int i = 0; i < rows; i++) {
            file >> layer.biases[i][0];
        }

        // добавляем слой в сеть
        layers.push_back(layer);
    }

    // закрываем файл
    file.close();
    log("Модель загружена\n");
    return true;
}