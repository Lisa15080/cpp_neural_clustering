// подключаем нашу нейросеть
#include "neural_net.h"
// подключаем для работы с русским языком в консоли
#include <windows.h>

using namespace std;

int main() {
    // настраиваем кодировку для русского текста в консоли
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    cout << "Тест\n";

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
        cout << "Модель сохранена в my_model.txt\n";
    }

    cout << "\nСоздаем новую сеть и загружаем сохраненную модель\n";

    // создаем вторую сеть (тоже с логированием, но в другой файл)
    NeuralNetwork net2({ 2, 3, 1 }, true, "network_log2.txt");

    net2.loadModel("model.txt");

    // проверяем что загрузилось
    cout << "Веса после загрузки:\n";
    net2.printLayers();

    return 0;
}