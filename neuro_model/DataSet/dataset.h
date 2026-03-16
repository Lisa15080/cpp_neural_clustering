#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <cstddef>
// Структура для хранения данных
struct Dataset {
    std::vector<std::vector<double>> inputs;   // Входные данные
    std::vector<std::vector<double>> targets;  // Правильные ответы
};

// Класс для генерации данных
class DatasetGenerator {
public:
    // Генерируем гауссовы кластеры
    // n_samples - сколько точек всего
    // cluster_std - разброс точек (чем больше, тем сложнее)
    // separation - расстояние между кластерами (чем меньше, тем сложнее)
    Dataset generate_gaussian(size_t n_samples = 100, 
                              double cluster_std = 0.5,
                              double separation = 2.0);
};

#endif