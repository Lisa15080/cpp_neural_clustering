#include "neural_net.h"
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

// Функции активации
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double NeuralNetwork::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double NeuralNetwork::linear(double x) {
    return x;
}

double NeuralNetwork::linearDerivative(double /*x*/) {
    return 1.0;
}

void NeuralNetwork::softmax(Matrix<double>& mat) {
    for (size_t col = 0; col < mat.cols(); ++col) {
        double maxVal = mat(0, col);
        for (size_t row = 1; row < mat.rows(); ++row)
            if (mat(row, col) > maxVal) maxVal = mat(row, col);
        double sumExp = 0.0;
        for (size_t row = 0; row < mat.rows(); ++row) {
            double val = exp(mat(row, col) - maxVal);
            mat(row, col) = val;
            sumExp += val;
        }
        for (size_t row = 0; row < mat.rows(); ++row)
            mat(row, col) /= sumExp;
    }
}

void NeuralNetwork::applyActivation(Matrix<double>& mat, Activation act) {
    if (act == Activation::SOFTMAX) {
        softmax(mat);
        return;
    }
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j) {
            double x = mat(i, j);
            switch (act) {
                case Activation::SIGMOID: mat(i, j) = sigmoid(x); break;
                case Activation::RELU:    mat(i, j) = relu(x);    break;
                case Activation::LINEAR:  mat(i, j) = linear(x);  break;
                default: break;
            }
        }
}

void NeuralNetwork::applyActivationDerivative(Matrix<double>& mat, Activation act) {
    if (act == Activation::SOFTMAX)
        throw runtime_error("Производная SOFTMAX не реализована напрямую.");
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j) {
            double x = mat(i, j);
            switch (act) {
                case Activation::SIGMOID: mat(i, j) = sigmoidDerivative(x); break;
                case Activation::RELU:    mat(i, j) = reluDerivative(x);    break;
                case Activation::LINEAR:  mat(i, j) = linearDerivative(x);  break;
                default: break;
            }
        }
}

// Логирование
void NeuralNetwork::log(const string& message, bool toConsole) {
    if (toConsole) cout << message;
    if (loggingEnabled && logFile.is_open()) logFile << message;
}

// Генерация весов
double NeuralNetwork::randomWeight() {
    uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

void NeuralNetwork::checkNetworkNotEmpty() const {
    if (layers.size() < 2)
        throw runtime_error("NeuralNetwork: для выполнения операции необходимо хотя бы два слоя.");
}

// Конструктор
NeuralNetwork::NeuralNetwork(const vector<int>& sizes,
                             Activation hiddenActivation,
                             bool enableLogging,
                             const string& filename)
    : loggingEnabled(enableLogging), logFilename(filename), rng(random_device{}())
{
    if (sizes.size() < 2)
        throw invalid_argument("NeuralNetwork: вектор sizes должен содержать минимум 2 значения.");

    if (loggingEnabled) {
        logFile.open(logFilename);
        log("Лог нейросети \nАрхитектура: ");
        for (int s : sizes) log(to_string(s) + " ");
        log("\n");
    }

    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        int inSize  = sizes[i];
        int outSize = sizes[i + 1];
        Layer layer;
        layer.weights = Matrix<double>(outSize, inSize);
        layer.biases  = Matrix<double>(outSize, 1);
        for (int r = 0; r < outSize; ++r) {
            for (int c = 0; c < inSize; ++c)
                layer.weights(r, c) = randomWeight();
            layer.biases(r, 0) = randomWeight();
        }
        layer.activation = (i == sizes.size() - 2) ? Activation::SIGMOID : hiddenActivation;
        layers.push_back(layer);
    }
    log("Сеть создана.\n\n");
}

NeuralNetwork::~NeuralNetwork() {
    if (logFile.is_open()) logFile.close();
}

// Добавление слоя
void NeuralNetwork::addLayer(int outputSize, Activation activation) {
    if (layers.empty())
        throw runtime_error("addLayer: нельзя добавить слой к пустой сети.");
    int inputSize = static_cast<int>(layers.back().weights.rows());
    Layer newLayer;
    newLayer.weights = Matrix<double>(outputSize, inputSize);
    newLayer.biases  = Matrix<double>(outputSize, 1);
    for (int r = 0; r < outputSize; ++r) {
        for (int c = 0; c < inputSize; ++c)
            newLayer.weights(r, c) = randomWeight();
        newLayer.biases(r, 0) = randomWeight();
    }
    newLayer.activation = activation;
    layers.push_back(newLayer);
    log("Добавлен слой: " + to_string(inputSize) + " -> " + to_string(outputSize) +
        " (активация " + to_string(static_cast<int>(activation)) + ")\n");
}

// Копирование весов
void NeuralNetwork::copyWeightsFrom(const NeuralNetwork& other) {
    if (layers.size() != other.layers.size())
        throw runtime_error("copyWeightsFrom: архитектуры не совпадают.");
    for (size_t l = 0; l < layers.size(); ++l) {
        if (layers[l].weights.rows() != other.layers[l].weights.rows() ||
            layers[l].weights.cols() != other.layers[l].weights.cols())
            throw runtime_error("copyWeightsFrom: несовпадение размеров на слое " + to_string(l));
        layers[l].weights = other.layers[l].weights;
        layers[l].biases  = other.layers[l].biases;
    }
}

// Прямой проход
vector<double> NeuralNetwork::forward(const vector<double>& input) {
    checkNetworkNotEmpty();
    if (input.size() != layers[0].weights.cols())
        throw runtime_error("forward: размер входа не совпадает.");
    Matrix<double> X(input.size(), 1);
    for (size_t i = 0; i < input.size(); ++i) X(i, 0) = input[i];
    Matrix<double> out = forwardBatch(X);
    vector<double> result(out.rows());
    for (size_t i = 0; i < out.rows(); ++i) result[i] = out(i, 0);
    return result;
}

Matrix<double> NeuralNetwork::forwardBatch(const Matrix<double>& X) {
    checkNetworkNotEmpty();
    if (X.rows() != layers[0].weights.cols())
        throw runtime_error("forwardBatch: неверное количество признаков.");
    Matrix<double> A = X;
    for (size_t l = 0; l < layers.size(); ++l) {
        Layer& layer = layers[l];
        Matrix<double> Z = layer.weights * A;
        for (size_t i = 0; i < Z.rows(); ++i)
            for (size_t j = 0; j < Z.cols(); ++j)
                Z(i, j) += layer.biases(i, 0);
        layer.z = Z;
        Matrix<double> A_next = Z;
        applyActivation(A_next, layer.activation);
        layer.a = A_next;
        A = A_next;
    }
    return A;
}

// Обратное распространение
void NeuralNetwork::backward(const vector<double>& x,
                             const vector<double>& y_true,
                             vector<Matrix<double>>& dW,
                             vector<Matrix<double>>& db) {
    checkNetworkNotEmpty();
    size_t L = layers.size();
    dW.resize(L);
    db.resize(L);
    if (y_true.size() != layers[L-1].a.rows())
        throw runtime_error("backward: размер целевого вектора не совпадает.");
    const Layer& last = layers[L-1];
    Matrix<double> delta(last.a.rows(), 1);
    if (last.activation == Activation::SOFTMAX || last.activation == Activation::SIGMOID) {
        for (size_t i = 0; i < delta.rows(); ++i)
            delta(i, 0) = last.a(i, 0) - y_true[i];
    } else {
        throw runtime_error("backward: неподдерживаемая активация выходного слоя.");
    }
    for (int l = static_cast<int>(L) - 1; l >= 0; --l) {
        const Layer& layer = layers[l];
        Matrix<double> A_prev(l == 0 ? x.size() : layers[l-1].a.rows(), 1);
        if (l == 0) {
            for (size_t i = 0; i < x.size(); ++i) A_prev(i, 0) = x[i];
        } else {
            A_prev = layers[l-1].a;
        }
        dW[l] = Matrix<double>(delta.rows(), A_prev.rows());
        for (size_t i = 0; i < delta.rows(); ++i)
            for (size_t j = 0; j < A_prev.rows(); ++j)
                dW[l](i, j) = delta(i, 0) * A_prev(j, 0);
        db[l] = delta;
        if (l > 0) {
            Matrix<double> W_T = layer.weights.transpose();
            Matrix<double> newDelta = W_T * delta;
            const Layer& prevLayer = layers[l-1];
            if (prevLayer.activation == Activation::SOFTMAX)
                throw runtime_error("backward: SOFTMAX на скрытом слое не поддерживается.");
            for (size_t i = 0; i < newDelta.rows(); ++i) {
                double z_val = prevLayer.z(i, 0);
                double deriv = 0.0;
                switch (prevLayer.activation) {
                    case Activation::SIGMOID: deriv = sigmoidDerivative(z_val); break;
                    case Activation::RELU:    deriv = reluDerivative(z_val);    break;
                    case Activation::LINEAR:  deriv = linearDerivative(z_val);  break;
                    default: throw runtime_error("backward: неподдерживаемая активация скрытого слоя.");
                }
                newDelta(i, 0) *= deriv;
            }
            delta = newDelta;
        }
    }
}

// Обновление весов
void NeuralNetwork::updateWeights(const vector<Matrix<double>>& dW,
                                  const vector<Matrix<double>>& db,
                                  double learningRate) {
    checkNetworkNotEmpty();
    if (dW.size() != layers.size() || db.size() != layers.size())
        throw runtime_error("updateWeights: размер списка градиентов не совпадает.");
    for (size_t l = 0; l < layers.size(); ++l) {
        for (size_t i = 0; i < layers[l].weights.rows(); ++i) {
            for (size_t j = 0; j < layers[l].weights.cols(); ++j)
                layers[l].weights(i, j) -= learningRate * dW[l](i, j);
            layers[l].biases(i, 0) -= learningRate * db[l](i, 0);
        }
    }
}

// Предсказание
double NeuralNetwork::predictProba(const vector<double>& input) {
    vector<double> out = forward(input);
    return out.size() == 1 ? out[0] : (out.size() > 1 ? out[1] : 0.0);
}

vector<double> NeuralNetwork::predictProbabilities(const vector<double>& input) {
    vector<double> out = forward(input);
    if (out.size() == 1) {
        double p1 = out[0];
        return {1.0 - p1, p1};
    }
    return out;
}

int NeuralNetwork::predict(const vector<double>& input) {
    vector<double> probs = predictProbabilities(input);
    if (probs.size() == 2)
        return probs[1] >= 0.5 ? 1 : 0;
    return static_cast<int>(max_element(probs.begin(), probs.end()) - probs.begin());
}

// Точность на датасете
double NeuralNetwork::accuracy(const Datasetpars<double>& data) {
    if (data.inputs.rows() == 0) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < data.inputs.rows(); ++i) {
        vector<double> x(data.inputs.cols());
        for (size_t j = 0; j < data.inputs.cols(); ++j)
            x[j] = data.inputs(i, j);
        int pred = predict(x);
        int true_label = static_cast<int>(data.targets(i, 0) + 0.5);
        if (pred == true_label) ++correct;
    }
    return static_cast<double>(correct) / data.inputs.rows();
}

// Сохранение и загрузка
bool NeuralNetwork::saveModel(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        log("Ошибка: не могу открыть файл для сохранения: " + filename + "\n");
        return false;
    }
    file << layers.size() << "\n";
    for (const auto& layer : layers) {
        file << layer.weights.rows() << " " << layer.weights.cols() << "\n";
        for (size_t i = 0; i < layer.weights.rows(); ++i) {
            for (size_t j = 0; j < layer.weights.cols(); ++j)
                file << layer.weights(i, j) << " ";
            file << "\n";
        }
        for (size_t i = 0; i < layer.biases.rows(); ++i)
            file << layer.biases(i, 0) << " ";
        file << "\n" << static_cast<int>(layer.activation) << "\n";
    }
    file.close();
    log("Модель сохранена: " + filename + "\n");
    return true;
}

bool NeuralNetwork::loadModel(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        log("Ошибка: не могу открыть файл для загрузки: " + filename + "\n");
        return false;
    }
    size_t numLayers;
    file >> numLayers;
    layers.clear();
    for (size_t l = 0; l < numLayers; ++l) {
        size_t rows, cols;
        file >> rows >> cols;
        Layer layer;
        layer.weights = Matrix<double>(rows, cols);
        layer.biases  = Matrix<double>(rows, 1);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                file >> layer.weights(i, j);
        for (size_t i = 0; i < rows; ++i)
            file >> layer.biases(i, 0);
        int act;
        file >> act;
        layer.activation = static_cast<Activation>(act);
        layers.push_back(layer);
    }
    file.close();
    log("Модель загружена: " + filename + "\n");
    return true;
}

// Печать структуры
void NeuralNetwork::printLayers() {
    string msg = "=== Структура сети ===\n";
    for (size_t l = 0; l < layers.size(); ++l) {
        msg += "Слой " + to_string(l) + ": ";
        switch (layers[l].activation) {
            case Activation::SIGMOID: msg += "SIGMOID"; break;
            case Activation::RELU:    msg += "RELU";    break;
            case Activation::LINEAR:  msg += "LINEAR";  break;
            case Activation::SOFTMAX: msg += "SOFTMAX"; break;
        }
        msg += " | веса " + to_string(layers[l].weights.rows()) + "x" +
               to_string(layers[l].weights.cols()) + "\n";
    }
    log(msg + "\n");
}