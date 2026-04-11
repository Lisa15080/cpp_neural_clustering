#include "../neuro_model/DataSet/dataset.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstddef>
#include <numbers>

// Функция для генерации случайного числа из нормального распределения
// Используем простую формулу (сумма равномерных случайных чисел)
double gaussian_random(double mean, double std_dev) {
    double sum = 0.0;
    // Суммируем 12 случайных чисел - получается похоже на гауссиану
    for (int i = 0; i < 12; i++) {
        sum += (double)rand() / RAND_MAX;
    }
    sum -= 6.0;  // Центрируем вокруг 0
    return mean + sum * std_dev;
}

Dataset DatasetGenerator::generate_gaussian(size_t n_samples, 
                                             double cluster_std,
                                             double separation) {
    Dataset data;
    
    // Инициализируем генератор случайных чисел
    srand(time(nullptr));
    
    size_t n_per_class = n_samples / 2;  // Половина точек в каждом кластере
    
    // Лямбда-функция для генерации одного кластера
    // Захватываем по ссылке: data, n_per_class, cluster_std
    auto generate_cluster = [&](double center_x, double center_y, double label) {
        for (size_t i = 0; i < n_per_class; i++) {
            double x = gaussian_random(center_x, cluster_std);
            double y = gaussian_random(center_y, cluster_std);
            data.inputs.push_back({x, y});
            data.targets.push_back({label});
        }
    };
    
    // Генерация двух кластеров 
    // Кластер 1 (класс 0): центр справа (separation/2, 0)
    generate_cluster(separation / 2.0, 0.0, 0.0);
    // Кластер 2 (класс 1): центр слева (-separation/2, 0)
    generate_cluster(-separation / 2.0, 0.0, 1.0);
    
    return data;
}