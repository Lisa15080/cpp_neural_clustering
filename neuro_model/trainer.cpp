#include "trainer.h"
#include <cmath>
#include <iomanip>

using namespace std;

// Конструктор
Trainer::Trainer(NeuralNetwork& net, const TrainingConfig& cfg)
    : network(net), config(cfg) {}

// ===== ОБРАТНЫЙ ПРОХОД (BACKPROPAGATION) =====
void Trainer::backward_pass(
    const vector<Matrix<double>>& layer_outputs,
    const vector<double>& target,
    vector<Matrix<double>>& deltas
) {
    size_t num_layers = network.getLayers().size();
    
    // Ошибка выходного слоя
    Matrix<double> output = layer_outputs.back();
    Matrix<double> output_delta(output.rows(), 1);
    
    for (size_t j = 0; j < output.rows(); j++) {
        double out_val = output(j, 0);
        double t = target[j];
        // Градиент: (выход - цель) * производная сигмоиды
        output_delta(j, 0) = (out_val - t) * network.sigmoidDerivative(out_val);
    }
    deltas[num_layers - 1] = output_delta;
    
    // Ошибка скрытых слоёв (идём с конца к началу)
    auto& layers = network.getLayers();
    
    for (int l = (int)num_layers - 2; l >= 0; l--) {
        Matrix<double> delta(layer_outputs[l + 1].rows(), 1);
        
        for (size_t j = 0; j < delta.rows(); j++) {
            double error_sum = 0.0;
            
            // Суммируем влияние следующего слоя
            for (size_t k = 0; k < deltas[l + 1].rows(); k++) {
                error_sum += layers[l + 1].weights(k, j) * deltas[l + 1](k, 0);
            }
            
            double a = layer_outputs[l + 1](j, 0);
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
    
    for (size_t l = 0; l < layers.size(); l++) {
        // Обновляем веса
        for (size_t j = 0; j < layers[l].weights.rows(); j++) {
            for (size_t k = 0; k < layers[l].weights.cols(); k++) {
                double gradient = deltas[l](j, 0) * layer_outputs[l](k, 0);
                layers[l].weights(j, k) -= config.learning_rate * gradient;
            }
        }
        
        // Обновляем смещения
        for (size_t j = 0; j < layers[l].biases.rows(); j++) {
            layers[l].biases(j, 0) -= config.learning_rate * deltas[l](j, 0);
        }
    }
}

// ===== ОБУЧЕНИЕ НА DATASET =====
void Trainer::train(const Dataset& data) {
    train(data.inputs, data.targets);
}

// ===== ОСНОВНОЙ МЕТОД ОБУЧЕНИЯ =====
void Trainer::train(
    const vector<vector<double>>& inputs,
    const vector<vector<double>>& targets
) {
    if (config.verbose) {
        cout << "\n===== НАЧАЛО ОБУЧЕНИЯ =====\n";
        cout << "Эпох: " << config.epochs << "\n";
        cout << "Скорость обучения: " << config.learning_rate << "\n";
        cout << "Примеров: " << inputs.size() << "\n\n";
    }
    
    auto& layers = network.getLayers();
    
    for (int epoch = 0; epoch < config.epochs; epoch++) {
        double total_error = 0.0;
        
        // Проходим по всем примерам
        for (size_t sample = 0; sample < inputs.size(); sample++) {
            // ===== ПРЯМОЙ ПРОХОД =====
            vector<Matrix<double>> layer_outputs;
            
            // Входной вектор -> матрица-столбец
            Matrix<double> current(inputs[sample].size(), 1);
            for (size_t j = 0; j < inputs[sample].size(); j++) {
                current(j, 0) = inputs[sample][j];
            }
            layer_outputs.push_back(current);
            
            // Прямой проход через все слои
            for (size_t l = 0; l < layers.size(); l++) {
                // Z = W * A + b
                Matrix<double> z = layers[l].weights * current;
                
                // Добавляем bias
                for (size_t j = 0; j < z.rows(); j++) {
                    z(j, 0) += layers[l].biases(j, 0);
                }
                
                // Применяем сигмоиду
                for (size_t j = 0; j < z.rows(); j++) {
                    z(j, 0) = network.sigmoid(z(j, 0));
                }
                
                layer_outputs.push_back(z);
                current = z;
            }
            
            // ===== ВЫЧИСЛЕНИЕ ОШИБКИ =====
            double error = 0.0;
            for (size_t j = 0; j < targets[sample].size(); j++) {
                double diff = current(j, 0) - targets[sample][j];
                error += diff * diff;  // MSE
            }
            total_error += error;
            
            // ===== ОБРАТНЫЙ ПРОХОД =====
            vector<Matrix<double>> deltas(layers.size());
            backward_pass(layer_outputs, targets[sample], deltas);
            
            // ===== ОБНОВЛЕНИЕ ВЕСОВ =====
            update_weights(layer_outputs, deltas);
        }
        
        // Вывод прогресса
        if (config.verbose && ((epoch + 1) % 100 == 0 || epoch == 0)) {
            double avg_error = total_error / inputs.size();
            cout << "Эпоха " << (epoch + 1) 
                 << " - Средняя ошибка: " << fixed << setprecision(6) << avg_error << "\n";
        }
    }
    
    if (config.verbose) {
        cout << "\n===== ОБУЧЕНИЕ ЗАВЕРШЕНО =====\n\n";
    }
}

// ===== ОЦЕНКА ТОЧНОСТИ =====
double Trainer::evaluate(const Dataset& data) {
    int correct = 0;
    
    for (size_t i = 0; i < data.inputs.size(); i++) {
        auto result = network.forward(data.inputs[i]);
        int predicted = (result[0] > 0.5) ? 1 : 0;
        int actual = (int)data.targets[i][0];
        
        if (predicted == actual) {
            correct++;
        }
    }
    
    return 100.0 * correct / data.inputs.size();
}