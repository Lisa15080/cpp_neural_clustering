#include "../Trainer_class/trainer.h"
#include "../../class/Matrix/matrix.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>
#include <numbers>
#include <functional>
#include <algorithm>

using namespace std;

namespace {

    // Копирование строки матрицы в столбец
    Matrix<double> extract_column(const Matrix<double>& m, size_t row) {
        Matrix<double> result(m.cols(), 1);
        for (size_t j = 0; j < m.cols(); ++j) {
            result(j, 0) = m(row, j);
        }
        return result;
    }

    // MSE ошибка для одного вектора
    double compute_mse(const Matrix<double>& output, const Matrix<double>& target) {
        double error = 0.0;
        for (size_t j = 0; j < output.rows(); ++j) {
            double diff = output(j, 0) - target(j, 0);
            error += diff * diff;
        }
        return error;
    }

    // Пороговая классификация (только для бинарного случая)
    int threshold(double value) {
        return (value > 0.5) ? 1 : 0;
    }
}

// Конструктор
Trainer::Trainer(NeuralNetwork& net, const TrainingConfig& cfg)
    : network(net), config(cfg) {}


// ===== ПРЯМОЙ ПРОХОД С ВОЗВРАТОМ ВСЕХ ПРОМЕЖУТОЧНЫХ ЗНАЧЕНИЙ =====
vector<Matrix<double>> Trainer::forward_pass(const Matrix<double>& input) const {
    auto& layers = network.getLayers();
    vector<Matrix<double>> layer_outputs;

    layer_outputs.push_back(input);
    Matrix<double> current = input;

    for (size_t l = 0; l < layers.size(); ++l) {
        if (current.rows() != layers[l].weights.cols()) {
            throw invalid_argument(
                "Forward pass layer " + to_string(l) +
                ": current rows (" + to_string(current.rows()) +
                ") != weights cols (" + to_string(layers[l].weights.cols()) + ")"
            );
        }

        Matrix<double> z = layers[l].weights * current;

        for (size_t j = 0; j < z.rows(); ++j) {
            z(j, 0) += layers[l].biases(j, 0);
        }
        layers[l].z = z;

        //  Применяем активацию, указанную в слое 
        Matrix<double> activated = z;
        network.applyActivation(activated, layers[l].activation);

        layers[l].a = activated;
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

    if (layer_outputs.size() != num_layers + 1) {
        throw invalid_argument("backward_pass: layer_outputs size mismatch");
    }

    auto& layers = network.getLayers();
    const Matrix<double>& output = layer_outputs.back();

    if (output.rows() != target.rows()) {
        throw invalid_argument(
            "backward_pass: output rows (" + to_string(output.rows()) +
            ") != target rows (" + to_string(target.rows()) + ")"
        );
    }

    // Градиент для выходного слоя
    const auto& last_layer = layers[num_layers - 1];
    Matrix<double> output_delta(output.rows(), 1);

    if (last_layer.activation == Activation::SIGMOID) {
        // Sigmoid + MSE (нужно учитывать производную)
        for (size_t j = 0; j < output.rows(); ++j) {
            double out_val = output(j, 0);
            double t = target(j, 0);

            double z = last_layer.z(j, 0);
            double deriv = network.sigmoidDerivative(z);

            output_delta(j, 0) = (out_val - t) * deriv;
        }
    }
    else {
        // Для остальных активаций
        for (size_t j = 0; j < output.rows(); ++j) {
            double out_val = output(j, 0);
            double t = target(j, 0);
            double deriv = 0.0;

            switch (last_layer.activation) {
                case Activation::RELU:
                    deriv = network.reluDerivative(last_layer.z(j, 0));
                    break;
                case Activation::LINEAR:
                    deriv = 1.0;
                    break;
                default:
                    deriv = 1.0;
            }

            output_delta(j, 0) = (out_val - t) * deriv;
        }
    }

    deltas[num_layers - 1] = output_delta;
    // Обратный проход по скрытым слоям
    for (int l = static_cast<int>(num_layers) - 2; l >= 0; --l) {
        const Matrix<double>& current_output = layer_outputs[l + 1];
        Matrix<double> delta(current_output.rows(), 1);

        for (size_t j = 0; j < delta.rows(); ++j) {
            double error_sum = 0.0;
            for (size_t k = 0; k < deltas[l + 1].rows(); ++k) {
                error_sum += layers[l + 1].weights(k, j) * deltas[l + 1](k, 0);
            }
            delta(j, 0) = error_sum;
        }

        // 👇 Применяем производную активации предыдущего слоя
        const auto& prevLayer = layers[l];
        Matrix<double> deriv = prevLayer.z;  // z с прямого прохода
        network.applyActivationDerivative(deriv, prevLayer.activation);
        
        for (size_t i = 0; i < delta.rows(); ++i) {
            delta(i, 0) *= deriv(i, 0);
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
        const Matrix<double>& layer_input = layer_outputs[l];

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

        for (size_t j = 0; j < layers[l].weights.rows(); ++j) {
            for (size_t k = 0; k < layers[l].weights.cols(); ++k) {
                double gradient = deltas[l](j, 0) * layer_input(k, 0);
                layers[l].weights(j, k) -= config.learning_rate * gradient;
            }
        }

        for (size_t j = 0; j < layers[l].biases.rows(); ++j) {
            layers[l].biases(j, 0) -= config.learning_rate * deltas[l](j, 0);
        }
    }
}


// ===== ОБУЧЕНИЕ НА ОДНОМ ПРИМЕРЕ =====
double Trainer::train_on_sample(const Matrix<double>& input, const Matrix<double>& target) {
    vector<Matrix<double>> layer_outputs = forward_pass(input);

    const Matrix<double>& output = layer_outputs.back();
    
    // Выбираем функцию потерь в зависимости от активации выхода
    double error = 0.0;
    error = compute_mse(output, target);

    vector<Matrix<double>> deltas(network.getLayers().size());
    backward_pass(layer_outputs, target, deltas);

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

    if (inputs.rows() != targets.rows()) {
        throw invalid_argument(
            "train: number of samples mismatch: inputs rows=" +
            to_string(inputs.rows()) + ", targets rows=" + to_string(targets.rows())
        );
    }

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        double total_error = 0.0;

        for (size_t sample = 0; sample < inputs.rows(); ++sample) {
            Matrix<double> input = extract_column(inputs, sample);
            Matrix<double> target = extract_column(targets, sample);

            total_error += train_on_sample(input, target);
        }

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

// ===== ОБУЧЕНИЕ НА ВЕКТОРАХ (ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ) =====
void Trainer::train(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
) {
    if (inputs.empty() || targets.empty()) {
        throw invalid_argument("train: inputs or targets is empty");
    }
    
    Matrix<double> inputs_mat(inputs.size(), inputs[0].size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs_mat(i, j) = inputs[i][j];
        }
    }
    
    Matrix<double> targets_mat(targets.size(), targets[0].size());
    for (size_t i = 0; i < targets.size(); ++i) {
        for (size_t j = 0; j < targets[i].size(); ++j) {
            targets_mat(i, j) = targets[i][j];
        }
    }
    
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
        Matrix<double> input = extract_column(inputs, i);
        vector<Matrix<double>> layer_outputs = forward_pass(input);
        const Matrix<double>& output = layer_outputs.back();

        // Для бинарной классификации (один выходной нейрон)
        if (output.cols() == 1 && output.rows() == 1) {
            double pred_value = output(0, 0);
            double target_value = targets(i, 0);

            // Бинарная классификация: порог 0.5
            int predicted = (pred_value > 0.5) ? 1 : 0;
            int actual = static_cast<int>(target_value + 0.5);

            if (predicted == actual) {
                correct++;
            }
        }
    }

    return 100.0 * correct / inputs.rows();
}

// ===== ОЦЕНКА ТОЧНОСТИ НА DATASET =====
double Trainer::evaluate(const Dataset& data) const {
    return evaluate(data.inputs, data.targets);
}

// ===== ОЦЕНКА ТОЧНОСТИ НА ВЕКТОРАХ =====
double Trainer::evaluate(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
) const {
    if (inputs.empty() || targets.empty()) {
        return 0.0;
    }
    
    Matrix<double> inputs_mat(inputs.size(), inputs[0].size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            inputs_mat(i, j) = inputs[i][j];
        }
    }
    
    Matrix<double> targets_mat(targets.size(), targets[0].size());
    for (size_t i = 0; i < targets.size(); ++i) {
        for (size_t j = 0; j < targets[i].size(); ++j) {
            targets_mat(i, j) = targets[i][j];
        }
    }
    
    return evaluate(inputs_mat, targets_mat);
}

// ===== МЕТОДЫ ПРЕДСКАЗАНИЯ =====

// Предсказание для одного примера (возвращает матрицу-столбец)
Matrix<double> Trainer::predict(const Matrix<double>& input) const {
    auto activations = forward_pass(input);
    return activations.back();
}

// Предсказание для одного примера (возвращает вектор вероятностей)
std::vector<double> Trainer::predict(const std::vector<double>& input) const {
    Matrix<double> input_mat(input.size(), 1);
    for (size_t i = 0; i < input.size(); ++i) {
        input_mat(i, 0) = input[i];
    }
    
    Matrix<double> output_mat = predict(input_mat);
    
    std::vector<double> result(output_mat.rows());
    for (size_t i = 0; i < output_mat.rows(); ++i) {
        result[i] = output_mat(i, 0);
    }
    return result;
}

// Предсказание класса для одного примера (возвращает индекс класса)
int Trainer::predict_class(const std::vector<double>& input) const {
    auto probs = predict(input);
    int max_idx = 0;
    double max_prob = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

// ===== ПРЕДСКАЗАНИЕ ДЛЯ НЕСКОЛЬКИХ ПРИМЕРОВ =====
Matrix<double> Trainer::predict_batch(const Matrix<double>& inputs) const {
    Matrix<double> results(inputs.rows(), 1);
    
    for (size_t i = 0; i < inputs.rows(); ++i) {
        Matrix<double> input = extract_column(inputs, i);
        Matrix<double> output = predict(input);
        results(i, 0) = output(0, 0);  // для бинарного случая
    }
    
    return results;
}

// ===== ВЫЧИСЛЕНИЕ ОШИБКИ =====
double Trainer::compute_loss(const Matrix<double>& inputs, const Matrix<double>& targets) const {
    if (inputs.rows() != targets.rows()) {
        throw invalid_argument("compute_loss: inputs and targets size mismatch");
    }
    
    double total_loss = 0.0;
    const auto& last_layer = network.getLayers().back();
    
    for (size_t i = 0; i < inputs.rows(); ++i) {
        Matrix<double> input = extract_column(inputs, i);
        Matrix<double> target = extract_column(targets, i);
        
        vector<Matrix<double>> layer_outputs = forward_pass(input);
        const Matrix<double>& output = layer_outputs.back();

        total_loss += compute_mse(output, target);
    }
    
    return total_loss / inputs.rows();
}