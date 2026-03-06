#include "dataset.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    
    // ===== Кластер 1 (класс 0) =====
    // Центр кластера справа: (separation/2, 0)
    double center1_x = separation / 2.0;
    double center1_y = 0.0;
    
    for (size_t i = 0; i < n_per_class; i++) {
        // Генерируем точку с гауссовым разбросом вокруг центра
        double x = gaussian_random(center1_x, cluster_std);
        double y = gaussian_random(center1_y, cluster_std);
        
        // Добавляем в датасет
        data.inputs.push_back({x, y});
        data.targets.push_back({0.0});  // Метка класса 0
    }
    
    // ===== Кластер 2 (класс 1) =====
    // Центр кластера слева: (-separation/2, 0)
    double center2_x = -separation / 2.0;
    double center2_y = 0.0;
    
    for (size_t i = 0; i < n_per_class; i++) {
        double x = gaussian_random(center2_x, cluster_std);
        double y = gaussian_random(center2_y, cluster_std);
        
        data.inputs.push_back({x, y});
        data.targets.push_back({1.0});  // Метка класса 1
    }
    
    return data;
}