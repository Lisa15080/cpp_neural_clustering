#ifndef NEURAL_NET_H
#define NEURAL_NET_H

// подключаем библиотеки для работы
#include <vector> // для использования векторов
#include <iostream> // для вывода в консоль
#include <fstream> // для работы с файлами
#include <string> // для работы со строками(имена файлов)

using namespace std;

// класс нейросети
class NeuralNetwork {
private:
    // структура одного слоя сети
    struct Layer {
        vector<vector<double>> weights; // веса(таблица чисел, которые настраиваются)
        vector<vector<double>> biases; // смещения(добавляются к каждому нейрону)
    };

    vector<Layer> layers;  // список всех слоев сети (сколько слоев - столько элементов)

    // Для логирования в файл
    bool loggingEnabled; // true - записываем логи в файл, false - только в консоль
    ofstream logFile; // файл, в который пишем логи
    string logFilename; // имя файла с логами

    // функция для записи сообщений
    void log(const string& message, bool toConsole = true);

public:
    // конструктор - создает сеть
    // sizes - массив с архитектурой(например {2,3,1})
    // enableLogging - включать ли запись в файл
    // logFile - имя файла для логов
    NeuralNetwork(const vector<int>& sizes, bool enableLogging = false, const string& logFile = "network_log.txt");

    // прямой проход(главная функция, которая считает ответ по входу)
    vector<double> forward(const vector<double>& input);

    // функции активации(превращают числа в удобный диапазон)
    double sigmoid(double x); // сигмоида (от 0 до 1)
    double sigmoidDerivative(double x); // производная сигмоиды (для обучения)
    double relu(double x); // ReLU (max(0,x))
    double reluDerivative(double x); // производная ReLU (для обучения)

    // печатает слоев
    void printLayers();

    // возвращает ссылку на слои (для обучения)
    vector<Layer>& getLayers() { return layers; }

    // сохраняет текущую модель в файл
    bool saveModel(const string& filename);

    // загружает модель из файла
    bool loadModel(const string& filename);
};

#endif