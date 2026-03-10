#include "neural_net.h"
#include "trainer.h"       
#include "dataset.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
    #include <direct.h>
    #include <windows.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

#include "../class/matrix.h"

using namespace std;

string getCurrentPath() {
    char buffer[1024];
#ifdef _WIN32
    if (_getcwd(buffer, sizeof(buffer)) != nullptr) {
        return string(buffer);
    }
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) {
        return string(buffer);
    }
#endif
    return ".";
}

int main() {
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif

    cout << "Тест нейросети\n";
    cout << "========================\n";

    // Генерация данных
    DatasetGenerator generator;
    Dataset data = generator.generate_gaussian(100, 0.5, 2.0);

    cout << "\nСгенерировано гауссовых кластеров: " << data.inputs.size() << " точек\n";

    // Данные для XOR
    vector<vector<double>> xor_inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
    vector<vector<double>> xor_targets = {{0}, {1}, {1}, {0}};

    // СЕТЬ 1: XOR 
    cout << "\n[1] Создаём сеть для XOR: 2→3→1\n";
    NeuralNetwork net({2, 3, 1}, true, "network_log.txt");

    cout << "\nРезультаты ДО обучения:\n";
    for (size_t i = 0; i < xor_inputs.size(); i++) {
        auto res = net.forward(xor_inputs[i]);
        cout << "  [" << xor_inputs[i][0] << "," << xor_inputs[i][1] 
             << "] → " << fixed << setprecision(4) << res[0] << "\n";
    }

    cout << "\nОбучаем сеть на XOR (1000 эпох):\n";
    TrainingConfig cfg_xor;
    cfg_xor.epochs = 1000;
    cfg_xor.learning_rate = 0.5;
    cfg_xor.verbose = true;
    
    Trainer trainer_xor(net, cfg_xor);
    trainer_xor.train(xor_inputs, xor_targets);

    cout << "\nРезультаты ПОСЛЕ обучения:\n";
    for (size_t i = 0; i < xor_inputs.size(); i++) {
        auto res = net.forward(xor_inputs[i]);
        cout << "  [" << xor_inputs[i][0] << "," << xor_inputs[i][1] 
             << "] → " << fixed << setprecision(4) << res[0] 
             << " (ожидаем: " << xor_targets[i][0] << ")\n";
    }

    // СЕТЬ 2: ГАУССОВЫ КЛАСТЕРЫ
    cout << "\n\n[2] Создаём сеть для гауссовых кластеров: 2→5→1\n";
    NeuralNetwork net2({2, 5, 1}, true, "network_log2.txt");
    
    cout << "Обучаем на гауссовых кластерах (500 эпох):\n";
    TrainingConfig cfg_gauss;
    cfg_gauss.epochs = 500;
    cfg_gauss.learning_rate = 0.1;
    cfg_gauss.verbose = true;
    
    Trainer trainer_gauss(net2, cfg_gauss);
    trainer_gauss.train(data.inputs, data.targets);

    // Проверка точности
    double accuracy = trainer_gauss.evaluate(data);
    cout << "\nТочность на гауссовых кластерах: " << fixed << setprecision(2) << accuracy << "%\n";

    // Сохранение/загрузка
    cout << "\n[3] Сохраняем модель...\n";
    if (net.saveModel("model.txt")) {
        cout << "Модель сохранена: " << getCurrentPath() << "/model.txt\n";
    }

    cout << "\nЗагружаем и проверяем:\n";
    NeuralNetwork net_loaded({2, 3, 1}, false);
    if (net_loaded.loadModel("model.txt")) {
        cout << "Модель загружена успешно!\n";
        cout << "Проверка на XOR:\n";
        for (size_t i = 0; i < xor_inputs.size(); i++) {
            auto res = net_loaded.forward(xor_inputs[i]);
            cout << "  [" << xor_inputs[i][0] << "," << xor_inputs[i][1] 
                 << "] → " << fixed << setprecision(4) << res[0] << "\n";
        }
    }

    cout << "\n========================\n";
    cout << "Работа завершена!\n";

    return 0;
}