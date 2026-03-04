#include "neural_net.h"
#include <windows.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    // настраиваем кодировку для русского текста в консоли
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    cout << "Тест нейросети\n";
    cout << "========================\n";

    // тест 1
    cout << "Создаем сеть с логированием в файл\n";

    // создаем сеть: 2 входа, 3 скрытых нейрона, 1 выход
    // включаем логирование в файл "network_log.txt"
    NeuralNetwork net({ 2, 3, 1 }, true, "network_log.txt");

    // печатаем начальные веса
    cout << "\nНачальные веса:\n";
    net.printLayers();

    cout << "\nВыполняем прямой проход\n";

    // тестовые данные для XOR
    vector<vector<double>> tests = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    // для каждого тестового примера
    for (const auto& test : tests) {
        // получаем ответ от нейросети
        auto result = net.forward(test);
        // выводим результат
        cout << "Вход: [" << test[0] << ", " << test[1] << "] - Выход: " << result[0] << endl;
    }

    // сохраняем текущее состояние сети в файл
    if (net.saveModel("model.txt")) {
        cout << "\nМодель сохранена в model.txt\n";
    }

    cout << "\nСоздаем новую сеть и загружаем сохраненную модель\n";

    // создаем вторую сеть
    NeuralNetwork net2({ 2, 3, 1 }, true, "network_log2.txt");

    if (net2.loadModel("model.txt")) {
        cout << "Модель загружена из model.txt\n";
    }

    // проверяем что загрузилось
    cout << "\nВеса после загрузки:\n";
    net2.printLayers();

    // Ручное обучение сети на XOR (так как нет метода train)
    cout << "\nОбучаем сеть на XOR (10 эпох с ручным обучением):\n";

    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };

    // Простейшее ручное обучение (имитация, так как у нас нет обратного распространения)
    // В реальности нужно реализовать метод train в классе NeuralNetwork
    cout << "Примечание: Для реального обучения нужно реализовать метод train\n";
    cout << "в классе NeuralNetwork с алгоритмом обратного распространения ошибки\n\n";

    cout << "Текущие результаты работы сети (без обучения):\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto result = net.forward(inputs[i]);
        cout << "Вход: [" << inputs[i][0] << ", " << inputs[i][1]
                  << "] - Выход: " << fixed << setprecision(6) << result[0]
                  << " (ожидаем: " << targets[i][0] << ")\n";
    }

    cout << "\nРабота с отдельными слоями через getLayers():\n";
    auto& layers = net.getLayers();
    cout << "Количество слоев: " << layers.size() << endl;

    for (size_t i = 0; i < layers.size(); ++i) {
        cout << "Слой " << i + 1 << ":\n";
        cout << "  Веса: " << layers[i].weights.size() << "x"
                  << (layers[i].weights.empty() ? 0 : layers[i].weights[0].size()) << endl;
        cout << "  Смещения: " << layers[i].biases.size() << "x"
                  << (layers[i].biases.empty() ? 0 : layers[i].biases[0].size()) << endl;
    }

    return 0;
}