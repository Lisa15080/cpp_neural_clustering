#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "../../class/Matrix/matrix.h"

using namespace std;

class NeuralNetwork {
private:
    struct Layer {
        Matrix<double> weights;
        Matrix<double> biases;
    };

    vector<Layer> layers;
    bool loggingEnabled;
    ofstream logFile;
    string logFilename;

    void log(const string& message, bool toConsole = true);

public:
    NeuralNetwork(const vector<int>& sizes, bool enableLogging = false,
                  const string& logFile = "network_log.txt");

    vector<double> forward(const vector<double>& input);

    double sigmoid(double x);
    double sigmoidDerivative(double x);
    double relu(double x);
    double reluDerivative(double x);

    void printLayers();
    vector<Layer>& getLayers() { return layers; }
    bool saveModel(const string& filename);
    bool loadModel(const string& filename);
 
};

#endif