#ifndef TRAINER_H
#define TRAINER_H

#include "../Neural_Net/neural_net.h"
#include "../DataSet/dataset.h"
#include "../../class/Matrix/matrix.h"
#include <vector>
#include <iostream>
#include <functional>

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

    // ========== ВНУТРЕННИЕ МЕТОДЫ ==========

    // Прямой проход с сохранением всех промежуточных значений
    std::vector<Matrix<double>> forward_pass(const Matrix<double>& input) const;

    // Обратный проход (backpropagation)
    void backward_pass(
        const std::vector<Matrix<double>>& layer_outputs,
        const Matrix<double>& target,
        std::vector<Matrix<double>>& deltas
    );

    // Обновление весов на основе дельт
    void update_weights(
        const std::vector<Matrix<double>>& layer_outputs,
        const std::vector<Matrix<double>>& deltas
    );

    // Обучение на одном примере (возвращает ошибку)
    double train_on_sample(const Matrix<double>& input, const Matrix<double>& target);

public:
    // ========== КОНСТРУКТОР ==========
    explicit Trainer(NeuralNetwork& net, const TrainingConfig& cfg = TrainingConfig());

    // ========== МЕТОДЫ ОБУЧЕНИЯ ==========

    // Обучение на матрицах
    void train(const Matrix<double>& inputs, const Matrix<double>& targets);

    // Обучение на данных из Dataset (конвертирует в матрицы)
    void train(const Dataset& data);

    // Обучение на отдельных векторах (для обратной совместимости)
    void train(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets
    );

    // ========== МЕТОДЫ ОЦЕНКИ ==========

    // Оценка точности на матрицах
    double evaluate(const Matrix<double>& inputs, const Matrix<double>& targets) const;

    // Оценка точности на Dataset
    double evaluate(const Dataset& data) const;

    // Оценка точности на векторах (для обратной совместимости)
    double evaluate(
        const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets
    ) const;

    // ========== МЕТОДЫ ПРЕДСКАЗАНИЯ ==========

    // Предсказание для одного примера (возвращает вектор вероятностей)
    std::vector<double> predict(const std::vector<double>& input) const;

    // Предсказание для одного примера (возвращает матрицу-столбец)
    Matrix<double> predict(const Matrix<double>& input) const;

    // 👇 Предсказание класса для одного примера (возвращает индекс класса)
    int predict_class(const std::vector<double>& input) const;

    // Предсказание для нескольких примеров (матрица входов -> матрица выходов)
    Matrix<double> predict_batch(const Matrix<double>& inputs) const;

    // ========== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ==========

    // Получить текущую ошибку на данных
    double compute_loss(const Matrix<double>& inputs, const Matrix<double>& targets) const;

    // Получить текущие настройки
    const TrainingConfig& getConfig() const { return config; }

    // Изменить настройки (например, скорость обучения во время тренировки)
    void setConfig(const TrainingConfig& new_config) { config = new_config; }
};

#endif