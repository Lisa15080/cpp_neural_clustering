#include "../Neural_Net/neural_net.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

void NeuralNetwork::log(const string& message, bool toConsole) {
    if (toConsole) cout << message;
    if (loggingEnabled && logFile.is_open()) logFile << message;
}

// Конструктор
NeuralNetwork::NeuralNetwork(const vector<int>& sizes, bool enableLogging, const string& filename) {
    srand(time(nullptr));

    loggingEnabled = enableLogging;
    logFilename = filename;

    if (loggingEnabled) {
        logFile.open(logFilename);
        logFile << "Лог нейросети" << endl;
        logFile << "Архитектура: ";
        for (int size : sizes) logFile << size << " ";
        logFile << endl << endl;
    }

    string archMsg = "Создаем нейросеть с архитектурой: ";
    for (int size : sizes) archMsg += to_string(size) + " ";
    log(archMsg + "\n");

    for (size_t i = 0; i < sizes.size() - 1; i++) {
        int inputSize = sizes[i];
        int outputSize = sizes[i + 1];

        string layerMsg = "Слой " + to_string(i) + ": " +
            to_string(inputSize) + " -> " +
            to_string(outputSize) + " нейронов\n";
        log(layerMsg);

        Layer layer;
        layer.weights = Matrix<double>(outputSize, inputSize);
        layer.biases = Matrix<double>(outputSize, 1);

        for (int j = 0; j < outputSize; j++) {
            for (int k = 0; k < inputSize; k++) {
                layer.weights(j, k) = ((double)rand() / RAND_MAX) * 2 - 1;
            }
            layer.biases(j, 0) = ((double)rand() / RAND_MAX) * 2 - 1;
        }

        layers.push_back(layer);
    }

    log("Нейросеть создана\n\n");
}

// Функции активации
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double NeuralNetwork::relu(double x) {
    return x > 0 ? x : 0;
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0 ? 1 : 0;
}

// Прямой проход
vector<double> NeuralNetwork::forward(const vector<double>& input) {
    string msg = "Прямой проход: входные данные = ";
    for (double val : input) msg += to_string(val) + " ";
    log(msg + "\n");

    Matrix<double> current(input.size(), 1);
    for (size_t i = 0; i < input.size(); i++) {
        current(i, 0) = input[i];
    }

    for (size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++) {
        const Layer& layer = layers[layerIdx];
        Matrix<double> next = layer.weights * current;

        for (size_t i = 0; i < next.rows(); i++) {
            next(i, 0) += layer.biases(i, 0);
            next(i, 0) = sigmoid(next(i, 0));
        }
        current = next;
    }

    vector<double> result(current.rows());
    for (size_t i = 0; i < current.rows(); i++) {
        result[i] = current(i, 0);
    }

    log("Результат: ");
    for (double val : result) log(to_string(val) + " ");
    log("\n------------------------\n");

    return result;
}


// Печать слоёв
void NeuralNetwork::printLayers() {
    string msg = "Структура нейросети:\n";
    log(msg);

    for (size_t l = 0; l < layers.size(); l++) {
        string layerMsg = "Слой " + to_string(l) + ":\n";
        log(layerMsg);

        log("  Веса:\n");
        for (size_t i = 0; i < layers[l].weights.rows(); i++) {
            string rowMsg = "    ";
            for (size_t j = 0; j < layers[l].weights.cols(); j++) {
                rowMsg += to_string(layers[l].weights(i, j)) + " ";
            }
            log(rowMsg + "\n");
        }

        log("  Смещения:\n    ");
        for (size_t i = 0; i < layers[l].biases.rows(); i++) {
            log(to_string(layers[l].biases(i, 0)) + " ");
        }
        log("\n");
    }
}

// Сохранение модели
bool NeuralNetwork::saveModel(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        log("Ошибка: Нельзя открыть файл для сохранения: " + filename + "\n");
        return false;
    }

    log("Сохраняем модель в файл: " + filename + "\n");

    file << layers.size() << endl;

    for (const auto& layer : layers) {
        file << layer.weights.rows() << " " << layer.weights.cols() << endl;

        for (size_t i = 0; i < layer.weights.rows(); i++) {
            for (size_t j = 0; j < layer.weights.cols(); j++) {
                file << layer.weights(i, j) << " ";
            }
            file << endl;
        }

        for (size_t i = 0; i < layer.biases.rows(); i++) {
            file << layer.biases(i, 0) << " ";
        }
        file << endl;
    }

    file.close();
    log("Модель сохранена\n");
    return true;
}

// Загрузка модели
bool NeuralNetwork::loadModel(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        log("Ошибка: Не могу открыть файл для загрузки: " + filename + "\n");
        return false;
    }

    log("Загружаем модель из файла: " + filename + "\n");

    int numLayers;
    file >> numLayers;
    layers.clear();

    for (int l = 0; l < numLayers; l++) {
        int rows, cols;
        file >> rows >> cols;

        Layer layer;
        layer.weights = Matrix<double>(rows, cols);
        layer.biases = Matrix<double>(rows, 1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file >> layer.weights(i, j);
            }
        }

        for (int i = 0; i < rows; i++) {
            file >> layer.biases(i, 0);
        }

        layers.push_back(layer);
    }

    file.close();
    log("Модель загружена\n");
    return true;
}