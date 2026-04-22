#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include "../../class/Matrix/matrix.h"   // путь к Matrix
#include "../../parser/pars.h"           // путь к парсеру

// Возможные функции активации
enum class Activation {
    SIGMOID,
    RELU,
    LINEAR,
    SOFTMAX
};

// Структура слоя
struct Layer {
    Matrix<double> weights;
    Matrix<double> biases;
    Activation activation;

    Matrix<double> z;   // взвешенная сумма
    Matrix<double> a;   // выход после активации
};

// Основной класс нейросети
class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& sizes,
                  Activation hiddenActivation = Activation::SIGMOID,
                  bool enableLogging = false,
                  const std::string& logFilename = "");

    ~NeuralNetwork();

    void addLayer(int outputSize, Activation activation = Activation::SIGMOID);

    size_t numLayers() const noexcept { return layers.size(); }
    size_t inputSize() const noexcept { return layers.empty() ? 0 : layers[0].weights.cols(); }
    size_t outputSize() const noexcept { return layers.empty() ? 0 : layers.back().weights.rows(); }

    std::vector<Layer>& getLayers() { return layers; }
    const std::vector<Layer>& getLayers() const { return layers; }

    std::vector<double> forward(const std::vector<double>& input);
    Matrix<double> forwardBatch(const Matrix<double>& X);

    void backward(const std::vector<double>& x,
                  const std::vector<double>& y_true,
                  std::vector<Matrix<double>>& dW,
                  std::vector<Matrix<double>>& db);

    void updateWeights(const std::vector<Matrix<double>>& dW,
                       const std::vector<Matrix<double>>& db,
                       double learningRate);

    double predictProba(const std::vector<double>& input);
    std::vector<double> predictProbabilities(const std::vector<double>& input);
    int predict(const std::vector<double>& input);

    double accuracy(const Datasetpars<double>& data);

    bool saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);

    void copyWeightsFrom(const NeuralNetwork& other);
    void printLayers();

    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    static double relu(double x);
    static double reluDerivative(double x);
    static double linear(double x);
    static double linearDerivative(double x);

    void applyActivation(Matrix<double>& mat, Activation act);
    void applyActivationDerivative(Matrix<double>& mat, Activation act);

private:
    std::vector<Layer> layers;

    void log(const std::string& message, bool toConsole = true);

    bool loggingEnabled;
    std::ofstream logFile;
    std::string logFilename;

    std::mt19937 rng;
    double randomWeight();

    void checkNetworkNotEmpty() const;
};

#endif
