#include "neural_net.h"
#include <windows.h>
#include <iostream>
#include <iomanip>
#include "../class/matrix.h"

using namespace std;
string getCurrentPath() {
    char buffer[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, buffer);
    return string(buffer);
}
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

    // Создаем матрицу 4x2 для тестовых данных
    Matrix<double> tests(4, 2);

    // Заполняем значениями
    tests(0, 0) = 0; tests(0, 1) = 0;  // {0, 0}
    tests(1, 0) = 0; tests(1, 1) = 1;  // {0, 1}
    tests(2, 0) = 1; tests(2, 1) = 0;  // {1, 0}
    tests(3, 0) = 1; tests(3, 1) = 1;  // {1, 1}

    // для каждого тестового примера
    for (int i = 0; i < tests.rows(); i++) {
        vector<double> test = {tests(i, 0), tests(i, 1)};
        auto result = net.forward(test);
        cout << "Вход: [" << tests(i, 0) << ", " << tests(i, 1) << "] - Выход: " << result[0] << endl;
    }
    // сохраняем текущее состояние сети в файл
    if (net.saveModel("model.txt")) {
        cout << "Полный путь: " << getCurrentPath() << "\\model.txt" << endl;
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
        cout << "  Веса: " << layers[i].weights.rows() << "x"
                  << layers[i].weights.cols() << endl;
        cout << "  Смещения: " << layers[i].biases.rows() << "x"
                  << layers[i].biases.cols() << endl;
    }

    return 0;
}