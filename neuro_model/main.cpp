// подключение заголовочных файлов нейронной сети, тренера, матриц
#include "Neural_Net/neural_net.h"
#include "Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"

// стандартные библиотеки
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#ifdef _WIN32
    #include <direct.h>
    #define getcwd _getcwd
#else
    #include <unistd.h>
#endif

using namespace std;

// вспомогательные функции

// возвращает текущую рабочую директорию (для отладки)
string getCurrentPath() {
    char buffer[1024];
#ifdef _WIN32
    if (_getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#else
    if (getcwd(buffer, sizeof(buffer)) != nullptr) return string(buffer);
#endif
    return ".";
}

// преобразование Matrix<double> в vector<vector<double>> (для совместимости)
vector<vector<double>> matrixToVector(const Matrix<double>& mat) {
    vector<vector<double>> vec(mat.rows(), vector<double>(mat.cols()));
    for (size_t i = 0; i < mat.rows(); ++i)
        for (size_t j = 0; j < mat.cols(); ++j)
            vec[i][j] = mat(i, j);
    return vec;
}

// обратное преобразование
Matrix<double> vectorToMatrix(const vector<vector<double>>& vec) {
    if (vec.empty()) return Matrix<double>();
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Matrix<double> mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = vec[i][j];
    return mat;
}

// структура для хранения параметров нормализации (среднее и стандартное отклонение)
struct NormalizationParams {
    vector<double> mean;
    vector<double> std;
};

// вычисляет среднее и стандартное отклонение по каждому признаку и нормализует входные данные
NormalizationParams fitNormalizeData(vector<vector<double>>& inputs) {
    NormalizationParams params;
    if (inputs.empty() || inputs[0].empty()) return params;

    size_t n_samples = inputs.size();
    size_t n_features = inputs[0].size();

    // среднее
    params.mean.assign(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            params.mean[j] += inputs[i][j];
    for (size_t j = 0; j < n_features; ++j)
        params.mean[j] /= n_samples;

    // стандартное отклонение
    params.std.assign(n_features, 0.0);
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            params.std[j] += (inputs[i][j] - params.mean[j]) * (inputs[i][j] - params.mean[j]);
    for (size_t j = 0; j < n_features; ++j) {
        params.std[j] = sqrt(params.std[j] / n_samples);
        if (params.std[j] < 1e-8) params.std[j] = 1.0;   // защита от деления на ноль
    }

    // нормализация (z-score)
    for (size_t i = 0; i < n_samples; ++i)
        for (size_t j = 0; j < n_features; ++j)
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];

    return params;
}

// нормализует данные с уже готовыми параметрами
void normalizeDataWithParams(vector<vector<double>>& inputs, const NormalizationParams& params) {
    for (size_t i = 0; i < inputs.size(); ++i)
        for (size_t j = 0; j < inputs[i].size(); ++j)
            inputs[i][j] = (inputs[i][j] - params.mean[j]) / params.std[j];
}

// преобразует строку CSV в вектор чисел, пропуская столбец-цель (target_idx)
vector<double> processDataRow(const vector<string>& row, int target_idx) {
    vector<double> features;
    for (size_t i = 0; i < row.size(); ++i) {
        if ((int)i == target_idx) continue;   // не добавляем целевую переменную в признаки
        try {
            features.push_back(stod(row[i]));   // преобразуем строку в double
        } catch (...) {
            features.push_back(0.0);            // при ошибке вставляем 0
        }
    }
    return features;
}

// вычисление F1-меры (среднее гармоническое между точностью и полнотой)
double calcF1(Trainer& trainer,
              const vector<vector<double>>& X,
              const vector<double>& y,
              double threshold = 0.5) {
    int tp = 0, fp = 0, fn = 0;   // True Positive, False Positive, False Negative

    for (size_t i = 0; i < X.size(); ++i) {
        vector<double> out = trainer.predict(X[i]);   // предсказание сети
        int pred = (out[0] > threshold) ? 1 : 0;
        int real = (y[i] > 0.5) ? 1 : 0;

        if (pred == 1 && real == 1) tp++;
        if (pred == 1 && real == 0) fp++;
        if (pred == 0 && real == 1) fn++;
    }

    double precision = (tp + fp == 0) ? 0.0 : (double)tp / (tp + fp);
    double recall    = (tp + fn == 0) ? 0.0 : (double)tp / (tp + fn);

    if (precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

// обучение и оценка на одном датасете

double runDataset(const string& train_file) {
    cout << "\n==============================\n";
    cout << "Dataset: " << train_file << "\n";

    // открываем файл
    ifstream train_stream(train_file);
    if (!train_stream.is_open()) {
        throw runtime_error("Не удалось открыть файл: " + train_file);
    }

    string line;
    // читаем первую строку - заголовки
    getline(train_stream, line);
    vector<string> headers;
    stringstream header_ss(line);
    string token;

    // определяем разделитель
    char delimiter = '\t';
    if (line.find(',') != string::npos && line.find('\t') == string::npos) {
        delimiter = ',';
    }

    // разбираем заголовки
    while (getline(header_ss, token, delimiter)) {
        headers.push_back(token);
    }

    // ищем столбец с целевой переменной
    int target_idx = -1;
    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == "target" || headers[i] == "Exited") {
            target_idx = i;
            break;
        }
    }
    // если не найден - считаем последний столбец целевым
    if (target_idx == -1 && !headers.empty()) {
        target_idx = headers.size() - 1;
        cout << "Столбец 'target' не найден, использую последний столбец как целевой\n";
    }

    // диагностическая информация
    cout << "Целевой столбец: " << headers[target_idx] << " (индекс " << target_idx << ")\n";

    vector<vector<double>> train_features;   // признаки (входы)
    vector<double> train_targets;            // целевые значения

    // построчное чтение данных
    while (getline(train_stream, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<string> row;
        while (getline(ss, token, delimiter)) {
            row.push_back(token);
        }
        if (row.size() < 2) continue;   // недостаточно колонок - пропускаем

        // извлекаем признаки (кроме целевого столбца)
        vector<double> features = processDataRow(row, target_idx);
        train_features.push_back(features);

        // извлекаем целевое значение
        if (target_idx >= 0 && target_idx < (int)row.size()) {
            try {
                train_targets.push_back(stod(row[target_idx]));
            } catch (...) {
                train_targets.push_back(0.0);
            }
        }
    }
    train_stream.close();

    if (train_features.empty()) {
        throw runtime_error("Не удалось загрузить данные из файла: " + train_file);
    }

    cout << "Загружено " << train_features.size()
         << " примеров, " << train_features[0].size() << " признаков\n";

    // подсчёт количества примеров каждого класса
    int count0 = 0, count1 = 0;
    for (double t : train_targets) {
        if (t < 0.5) count0++;
        else count1++;
    }
    cout << "Класс 0: " << count0 << ", класс 1: " << count1 << "\n";
    if (count0 == 0 || count1 == 0) {
        cout << "Внимание: Только один класс в данных\n";
    }

    // нормализация данных (z-score)
    cout << "Нормализация данных";
    fitNormalizeData(train_features);

    // перемешивание данных (фиксированный seed для воспроизводимости)
    vector<size_t> idx(train_features.size());
    iota(idx.begin(), idx.end(), 0);
    mt19937 gen(42);
    shuffle(idx.begin(), idx.end(), gen);

    vector<vector<double>> shuffled_features;
    vector<double> shuffled_targets;
    for (size_t id : idx) {
        shuffled_features.push_back(train_features[id]);
        shuffled_targets.push_back(train_targets[id]);
    }
    train_features = shuffled_features;
    train_targets = shuffled_targets;

    // разделение на тренировочную (80%) и валидационную (20%) выборки
    size_t total_samples = train_features.size();
    size_t val_size = total_samples / 5;          // 20%
    size_t train_size = total_samples - val_size; // 80%

    vector<vector<double>> train_feat(train_features.begin(), train_features.begin() + train_size);
    vector<double> train_targ(train_targets.begin(), train_targets.begin() + train_size);
    vector<vector<double>> val_feat(train_features.begin() + train_size, train_features.end());
    vector<double> val_targ(train_targets.begin() + train_size, train_targets.end());

    // формирование целевых векторов для тренера (ожидает вектор векторов)
    vector<vector<double>> train_targets_vec;
    for (double t : train_targ) {
        train_targets_vec.push_back({t});
    }

    cout << "Разделение данных:";
    cout << "Train samples: " << train_feat.size() << "\n";
    cout << "Validation samples: " << val_feat.size() << "\n";

    size_t n_features = train_feat[0].size();

    // создание нейронной сети: количество нейронов на входе равно числу признаков,
    // затем два скрытых слоя (16 и 8 нейронов) и выходной слой с 1 нейроном (бинарная классификация)
    cout << "Создание сети: "
         << n_features << " - 16 - 8 - 1\n";
    NeuralNetwork net({(int)n_features, 16, 8, 1}, Activation::RELU, true, "log.txt");

    // конфигурация обучения
    TrainingConfig cfg;
    cfg.epochs = 200;           // количество эпох
    cfg.learning_rate = 0.01;   // скорость обучения
    cfg.verbose = true;          // выводить прогресс

    cout << "Обучение";
    Trainer trainer(net, cfg);
    trainer.train(train_feat, train_targets_vec);

    // оценка качества на тренировочной и валидационной выборках
    double train_f1 = calcF1(trainer, train_feat, train_targ, 0.5);
    double val_f1   = calcF1(trainer, val_feat, val_targ, 0.5);

    cout << "Результаты:";
    cout << "F1 на train:      " << fixed << setprecision(4) << train_f1 << "\n";
    cout << "F1 на validation: " << fixed << setprecision(4) << val_f1 << "\n";

    return val_f1;   // возвращаем F1 на валидации (основная метрика)
}

// главная функция

int main() {
#ifdef _WIN32
    system("chcp 65001 > nul");   // установка кодировки UTF-8 для Windows
#endif

    cout << "Нейронная сеть. Проверка F1 на датасетах";
    cout << "Текущая папка: " << getCurrentPath() << "\n";

    try {
        // Список путей к датасетам (файлы должны лежать в папке datasets)
        vector<string> datasets = {
            "datasets/dataset1.csv",   // первый датасет (2 признака)
            "datasets/dataset2.csv",   // второй датасет (4 признака)
        };

        // предварительная проверка существования файлов
        for (const auto& file : datasets) {
            ifstream test(file);
            if (!test.is_open()) {
                cout << "Файл не найден: " << file << "\n";
            } else {
                test.close();
            }
        }

        vector<double> scores;

        // запуск обработки каждого существующего датасета
        for (const string& file : datasets) {
            ifstream check(file);
            if (!check.is_open()) {
                cout << "\nПропускаем файл: " << file << " (не найден)\n";
                continue;
            }
            check.close();
            double f1 = runDataset(file);
            scores.push_back(f1);
        }

        // вычисление итогового скора в зависимости от количества обработанных датасетов
        if (scores.size() == 3) {
            double final_score = (scores[0] + scores[1] + scores[2]) / 3.0;
            cout << "\n==============================\n";
            cout << "Итоговый score\n";
            cout << "F1(d1): " << fixed << setprecision(4) << scores[0] << "\n";
            cout << "F1(d2): " << fixed << setprecision(4) << scores[1] << "\n";
            cout << "Среднее F1 = " << fixed << setprecision(4) << final_score << "\n";
        }
        else if (scores.size() == 2) {
            double final_score = 0.5 * scores[0] + 0.5 * scores[1];
            cout << "\n==============================\n";
            cout << "Итоговый score\n";
            cout << "F1(d1): " << fixed << setprecision(4) << scores[0] << "\n";
            cout << "F1(d2): " << fixed << setprecision(4) << scores[1] << "\n";
            cout << "0.5 * F1(d1) + 0.5 * F1(d2) = " << fixed << setprecision(4) << final_score << "\n";
        }
        else {
            cout << "\nЗагружено " << scores.size() << " датасетов из " << datasets.size() << "\n";
        }

    } catch (const exception& e) {
        cerr << "\nОшибка: " << e.what() << "\n";
        return 1;
    }
}