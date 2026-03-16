#include "../Trainer_class/trainer.h"
#include "../../class/Matrix/matrix.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

// Конструктор
Trainer::Trainer(NeuralNetwork& net, const TrainingConfig& cfg)
    : network(net), config(cfg) {}

// ===== ПРЯМОЙ ПРОХОД С ВОЗВРАТОМ ВСЕХ ПРОМЕЖУТОЧНЫХ ЗНАЧЕНИЙ =====
vector<Matrix<double>> Trainer::forward_pass(const Matrix<double>& input) const {
    auto& layers = network.getLayers();
    vector<Matrix<double>> layer_outputs;

    // Входной слой (сохраняем вход)
    layer_outputs.push_back(input);

    // Текущий выход (начинаем с входа)
    Matrix<double> current = input;

    // Проходим через все слои
    for (size_t l = 0; l < layers.size(); ++l) {
        // Проверяем совместимость размеров
        if (current.rows() != layers[l].weights.cols()) {
            throw invalid_argument(
                "Forward pass layer " + to_string(l) +
                ": current rows (" + to_string(current.rows()) +
                ") != weights cols (" + to_string(layers[l].weights.cols()) + ")"
            );
        }

        // Z = W * A + b
        // weights: [output_size x input_size]
        // current: [input_size x 1]
        // result:  [output_size x 1]
        Matrix<double> z = layers[l].weights * current;

        // Добавляем bias
        for (size_t j = 0; j < z.rows(); ++j) {
            z(j, 0) += layers[l].biases(j, 0);
        }

        // Применяем функцию активации (сигмоиду)
        Matrix<double> activated(z.rows(), 1);
        for (size_t j = 0; j < z.rows(); ++j) {
            activated(j, 0) = network.sigmoid(z(j, 0));
        }

        layer_outputs.push_back(activated);
        current = activated;
    }

    return layer_outputs;
}

// ===== ОБРАТНЫЙ ПРОХОД (BACKPROPAGATION) =====
void Trainer::backward_pass(
    const vector<Matrix<double>>& layer_outputs,
    const Matrix<double>& target,
    vector<Matrix<double>>& deltas
) {
    size_t num_layers = network.getLayers().size();

    // Проверка размеров
    if (layer_outputs.size() != num_layers + 1) {
        throw invalid_argument("backward_pass: layer_outputs size mismatch");
    }

    auto& layers = network.getLayers();

    // ===== 1. ОШИБКА ВЫХОДНОГО СЛОЯ =====
    const Matrix<double>& output = layer_outputs.back();  // [output_size x 1]

    if (output.rows() != target.rows()) {
        throw invalid_argument(
            "backward_pass: output rows (" + to_string(output.rows()) +
            ") != target rows (" + to_string(target.rows()) + ")"
        );
    }

    // Дельта для выходного слоя: [output_size x 1]
    Matrix<double> output_delta(output.rows(), 1);

    for (size_t j = 0; j < output.rows(); ++j) {
        double out_val = output(j, 0);
        double t = target(j, 0);
        // Градиент: (выход - цель) * производная сигмоиды
        output_delta(j, 0) = (out_val - t) * network.sigmoidDerivative(out_val);
    }
    deltas[num_layers - 1] = output_delta;

    // ===== 2. ОШИБКА СКРЫТЫХ СЛОЁВ =====
    for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l) {
        // Выход текущего слоя: layer_outputs[l+1] - это активация после слоя l
        const Matrix<double>& current_output = layer_outputs[l + 1];  // [current_size x 1]

        // Дельта для текущего слоя: [current_size x 1]
        Matrix<double> delta(current_output.rows(), 1);

        for (size_t j = 0; j < delta.rows(); ++j) {
            double error_sum = 0.0;

            // Суммируем влияние от следующего слоя
            // layers[l+1].weights: [next_size x current_size]
            // deltas[l+1]: [next_size x 1]
            for (size_t k = 0; k < deltas[l + 1].rows(); ++k) {
                error_sum += layers[l + 1].weights(k, j) * deltas[l + 1](k, 0);
            }

            double a = current_output(j, 0);
            delta(j, 0) = error_sum * network.sigmoidDerivative(a);
        }
        deltas[l] = delta;
    }
}

// ===== ОБНОВЛЕНИЕ ВЕСОВ =====
void Trainer::update_weights(
    const vector<Matrix<double>>& layer_outputs,
    const vector<Matrix<double>>& deltas
) {
    auto& layers = network.getLayers();

    for (size_t l = 0; l < layers.size(); ++l) {
        // Вход для этого слоя - это выход предыдущего слоя
        const Matrix<double>& layer_input = layer_outputs[l];  // [input_size x 1]

        // Проверяем совместимость размеров
        if (deltas[l].rows() != layers[l].weights.rows()) {
            throw invalid_argument(
                "update_weights layer " + to_string(l) +
                ": deltas rows (" + to_string(deltas[l].rows()) +
                ") != weights rows (" + to_string(layers[l].weights.rows()) + ")"
            );
        }

        if (layer_input.rows() != layers[l].weights.cols()) {
            throw invalid_argument(
                "update_weights layer " + to_string(l) +
                ": input rows (" + to_string(layer_input.rows()) +
                ") != weights cols (" + to_string(layers[l].weights.cols()) + ")"
            );
        }

        // ===== ОБНОВЛЯЕМ ВЕСА =====
        // weights: [output_size x input_size]
        // deltas[l]: [output_size x 1]
        // layer_input: [input_size x 1]
        for (size_t j = 0; j < layers[l].weights.rows(); ++j) {        // j = output neuron
            for (size_t k = 0; k < layers[l].weights.cols(); ++k) {    // k = input neuron
                double gradient = deltas[l](j, 0) * layer_input(k, 0);
                layers[l].weights(j, k) -= config.learning_rate * gradient;
            }
        }

        // ===== ОБНОВЛЯЕМ СМЕЩЕНИЯ =====
        // biases: [output_size x 1]
        for (size_t j = 0; j < layers[l].biases.rows(); ++j) {
            layers[l].biases(j, 0) -= config.learning_rate * deltas[l](j, 0);
        }
    }
}

// ===== ОБУЧЕНИЕ НА ОДНОМ ПРИМЕРЕ =====
double Trainer::train_on_sample(const Matrix<double>& input, const Matrix<double>& target) {
    // Прямой проход
    vector<Matrix<double>> layer_outputs = forward_pass(input);

    // Вычисление ошибки (MSE)
    const Matrix<double>& output = layer_outputs.back();
    double error = 0.0;
    for (size_t j = 0; j < output.rows(); ++j) {
        double diff = output(j, 0) - target(j, 0);
        error += diff * diff;
    }

    // Обратный проход
    vector<Matrix<double>> deltas(network.getLayers().size());
    backward_pass(layer_outputs, target, deltas);

    // Обновление весов
    update_weights(layer_outputs, deltas);

    return error;
}

// ===== ОСНОВНОЙ МЕТОД ОБУЧЕНИЯ (ВЕРСИЯ С MATRIX) =====
void Trainer::train(const Matrix<double>& inputs, const Matrix<double>& targets) {
    if (config.verbose) {
        cout << "\n===== НАЧАЛО ОБУЧЕНИЯ =====\n";
        cout << "Эпох: " << config.epochs << "\n";
        cout << "Скорость обучения: " << config.learning_rate << "\n";
        cout << "Примеров: " << inputs.rows() << "\n";
        cout << "Размер входа: " << inputs.cols() << "\n";
        cout << "Размер выхода: " << targets.cols() << "\n\n";
    }

    // Проверка размеров
    if (inputs.rows() != targets.rows()) {
        throw invalid_argument(
            "train: number of samples mismatch: inputs rows=" +
            to_string(inputs.rows()) + ", targets rows=" + to_string(targets.rows())
        );
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        double total_error = 0.0;

        // Проходим по всем примерам
        for (size_t sample = 0; sample < inputs.rows(); ++sample) {
            // Извлекаем вход и цель для текущего примера
            Matrix<double> input(inputs.cols(), 1);
            Matrix<double> target(targets.cols(), 1);

            for (size_t j = 0; j < inputs.cols(); ++j) {
                input(j, 0) = inputs(sample, j);
            }
            for (size_t j = 0; j < targets.cols(); ++j) {
                target(j, 0) = targets(sample, j);
            }

            // Обучаем на одном примере
            total_error += train_on_sample(input, target);
        }

        // Вывод прогресса
        if (config.verbose && ((epoch + 1) % 100 == 0 || epoch == 0 || epoch == config.epochs - 1)) {
            double avg_error = total_error / inputs.rows();
            cout << "Эпоха " << (epoch + 1)
                 << " - Средняя ошибка: " << fixed << setprecision(6) << avg_error << "\n";
        }
    }

    if (config.verbose) {
        cout << "\n===== ОБУЧЕНИЕ ЗАВЕРШЕНО =====\n\n";
    }
}

// ===== ОБУЧЕНИЕ С ВЕКТОРАМИ (ДЛЯ СОВМЕСТИМОСТИ) =====
void Trainer::train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets) {
    if (inputs.empty() || targets.empty()) {
        throw invalid_argument("train: empty inputs or targets");
    }

    // Конвертируем векторы в матрицы
    Matrix<double> inputs_mat(inputs.size(), inputs[0].size());
    Matrix<double> targets_mat(targets.size(), targets[0].size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != inputs[0].size()) {
            throw invalid_argument("train: inconsistent input sizes");
        }
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs_mat(i, j) = inputs[i][j];
        }
    }

    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].size() != targets[0].size()) {
            throw invalid_argument("train: inconsistent target sizes");
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            targets_mat(i, j) = targets[i][j];
        }
    }

    // Вызываем матричную версию
    train(inputs_mat, targets_mat);
}

// ===== ОБУЧЕНИЕ НА DATASET =====
void Trainer::train(const Dataset& data) {
    train(data.inputs, data.targets);
}

// ===== ОЦЕНКА ТОЧНОСТИ (ВЕРСИЯ С MATRIX) =====
double Trainer::evaluate(const Matrix<double>& inputs, const Matrix<double>& targets) const {
    if (inputs.rows() != targets.rows()) {
        throw invalid_argument(
            "evaluate: number of samples mismatch: inputs rows=" +
            to_string(inputs.rows()) + ", targets rows=" + to_string(targets.rows())
        );
    }

    int correct = 0;

    for (size_t i = 0; i < inputs.rows(); ++i) {
        // Извлекаем вход
        Matrix<double> input(inputs.cols(), 1);
        for (size_t j = 0; j < inputs.cols(); ++j) {
            input(j, 0) = inputs(i, j);
        }

        // Прямой проход
        vector<Matrix<double>> layer_outputs = forward_pass(input);
        const Matrix<double>& output = layer_outputs.back();

        // Определяем предсказанный класс
        int predicted = (output(0, 0) > 0.5) ? 1 : 0;
        int actual = static_cast<int>(targets(i, 0));

        if (predicted == actual) {
            correct++;
        }
    }

    return 100.0 * correct / inputs.rows();
}

// ===== ОЦЕНКА ТОЧНОСТИ С ВЕКТОРАМИ =====
double Trainer::evaluate(const vector<vector<double>>& inputs, const vector<vector<double>>& targets) const {
    if (inputs.empty() || targets.empty()) {
        throw invalid_argument("evaluate: empty inputs or targets");
    }

    Matrix<double> inputs_mat(inputs.size(), inputs[0].size());
    Matrix<double> targets_mat(targets.size(), targets[0].size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs_mat(i, j) = inputs[i][j];
        }
    }

    for (size_t i = 0; i < targets.size(); ++i) {
        targets_mat(i, 0) = targets[i][0];
    }

    return evaluate(inputs_mat, targets_mat);
}

// ===== ОЦЕНКА ТОЧНОСТИ НА DATASET =====
double Trainer::evaluate(const Dataset& data) const {
    return evaluate(data.inputs, data.targets);
}

// ===== ПРЕДСКАЗАНИЕ ДЛЯ ОДНОГО ПРИМЕРА (ВОЗВРАЩАЕТ ВЕКТОР) =====
vector<double> Trainer::predict(const vector<double>& input) const {
    Matrix<double> input_mat(input.size(), 1);
    for (size_t j = 0; j < input.size(); ++j) {
        input_mat(j, 0) = input[j];
    }

    vector<Matrix<double>> layer_outputs = forward_pass(input_mat);
    const Matrix<double>& output = layer_outputs.back();

    vector<double> result(output.rows());
    for (size_t j = 0; j < output.rows(); ++j) {
        result[j] = output(j, 0);
    }

    return result;
}

// ===== ПРЕДСКАЗАНИЕ ДЛЯ ОДНОГО ПРИМЕРА (ВОЗВРАЩАЕТ МАТРИЦУ) =====
Matrix<double> Trainer::predict(const Matrix<double>& input) const {
    vector<Matrix<double>> layer_outputs = forward_pass(input);
    return layer_outputs.back();
}

// ===== ПРЕДСКАЗАНИЕ ДЛЯ МАТРИЦЫ ВХОДОВ =====
Matrix<double> Trainer::predict_batch(const Matrix<double>& inputs) const {
    Matrix<double> results(inputs.rows(), 1);

    for (size_t i = 0; i < inputs.rows(); ++i) {
        Matrix<double> input(inputs.cols(), 1);
        for (size_t j = 0; j < inputs.cols(); ++j) {
            input(j, 0) = inputs(i, j);
        }

        vector<Matrix<double>> layer_outputs = forward_pass(input);
        const Matrix<double>& output = layer_outputs.back();

        results(i, 0) = output(0, 0);
    }

    return results;
}

// ===== ВЫЧИСЛЕНИЕ ОШИБКИ =====
double Trainer::compute_loss(const Matrix<double>& inputs, const Matrix<double>& targets) const {
    if (inputs.rows() != targets.rows()) {
        throw invalid_argument(
            "compute_loss: number of samples mismatch: inputs rows=" +
            to_string(inputs.rows()) + ", targets rows=" + to_string(targets.rows())
        );
    }

    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.rows(); ++i) {
        Matrix<double> input(inputs.cols(), 1);
        for (size_t j = 0; j < inputs.cols(); ++j) {
            input(j, 0) = inputs(i, j);
        }

        vector<Matrix<double>> layer_outputs = forward_pass(input);
        const Matrix<double>& output = layer_outputs.back();

        for (size_t j = 0; j < output.rows(); ++j) {
            double diff = output(j, 0) - targets(i, j);
            total_loss += diff * diff;
        }
    }

    return total_loss / inputs.rows();
}