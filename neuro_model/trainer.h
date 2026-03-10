#ifndef TRAINER_H
#define TRAINER_H

#include "neural_net.h"
#include "dataset.h"
#include <vector>
#include <iostream>

// Настройки обучения
struct TrainingConfig {
    int epochs = 1000;              // Количество эпох
    double learning_rate = 0.1;     // Скорость обучения
    bool verbose = true;            // Выводить ли прогресс
};

// Класс для обучения нейросети
class Trainer {
private:
    NeuralNetwork& network;         // Ссылка на сеть которую обучаем
    TrainingConfig config;          // Настройки
    
    // Вспомогательные методы (внутренняя логика)
    void backward_pass(
        const std::vector<Matrix<double>>& layer_outputs,
        const std::vector<double>& target,
        std::vector<Matrix<double>>& deltas
    );
    
    void update_weights(
        const std::vector<Matrix<double>>& layer_outputs,
        const std::vector<Matrix<double>>& deltas
    );
    
public:
    // Конструктор
    Trainer(NeuralNetwork& net, const TrainingConfig& cfg = TrainingConfig());
    
    // Обучение на данных из Dataset
    void train(const Dataset& data);
    
    // Обучение на отдельных векторах
    void train(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets
    );
    
    // Оценка точности
    double evaluate(const Dataset& data);
};

#endif