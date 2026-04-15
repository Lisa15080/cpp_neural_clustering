# Trainer_class — Инструкция по работе

- **Папка:** `Trainer_class/`
- **Файлы:** `trainer.h`, `trainer.cpp`
- **Назначение:** Обучение, оценка и предсказание для нейронной сети `NeuralNetwork` с использованием backpropagation и MSE-потерь.

## ПОДКЛЮЧЕНИЕ

```cpp
#include "../Trainer_class/trainer.h"
```

## СТРУКТУРЫ ДАННЫХ

### TrainingConfig — настройки обучения

```cpp
struct TrainingConfig {
    int epochs = 1000;              // Количество эпох обучения
    double learning_rate = 0.1;     // Шаг градиентного спуска
    bool verbose = true;            // Выводить ли логи в консоль
};
```

### Dataset — контейнер данных (из dataset.h)

```cpp
template<typename T>
struct Dataset {
    Matrix<T> inputs;               // Признаки [n_samples × n_features]
    Matrix<T> targets;              // Цели [n_samples × n_targets]
    std::vector<std::string> headers; // Заголовки колонок
};
```

## КЛАСС Trainer

### Конструктор

```cpp
explicit Trainer(NeuralNetwork& net, const TrainingConfig& cfg = TrainingConfig());
```

| Параметр | Описание |
| :--- | :--- |
| `net` | Ссылка на обучаемую нейросеть (`NeuralNetwork`) |
| `cfg` | Конфигурация обучения (по умолчанию: 1000 эпох, lr=0.1, verbose=true) |


## МЕТОДЫ ОБУЧЕНИЯ

### train() — основной метод обучения

Поддерживает три перегрузки:

```cpp
// 1. Обучение на матрицах
void train(const Matrix<double>& inputs, const Matrix<double>& targets);

// 2. Обучение на Dataset
void train(const Dataset& data);

// 3. Обучение на векторах (обратная совместимость)
void train(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
);
```

| Параметр | Описание |
| :--- | :--- |
| `inputs` | Матрица признаков `[n_samples × n_features]` |
| `targets` | Матрица целей `[n_samples × n_targets]` |
| `data` | Объект `Dataset` с полями `inputs` и `targets` |

## МЕТОДЫ ОЦЕНКИ

### evaluate() — точность классификации (в %)

```cpp
// Для матриц
double evaluate(const Matrix<double>& inputs, const Matrix<double>& targets) const;

// Для Dataset
double evaluate(const Dataset& data) const;

// Для векторов
double evaluate(
    const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets
) const;
```
- Работает для бинарной классификации (порог 0.5).
- Возвращает процент правильно классифицированных примеров.

### compute_loss() — средняя MSE-ошибка

```cpp
double compute_loss(const Matrix<double>& inputs, const Matrix<double>& targets) const;
```
- Возвращает среднюю квадратичную ошибку на всём наборе данных.
- Полезно для мониторинга переобучения.

## МЕТОДЫ ПРЕДСКАЗАНИЯ

### predict() — получение выхода сети

```cpp
// Вход: вектор → выход: вектор вероятностей
std::vector<double> predict(const std::vector<double>& input) const;

// Вход: матрица-столбец → выход: матрица-столбец
Matrix<double> predict(const Matrix<double>& input) const;
```

**Пример:**
```cpp
std::vector<double> sample = {0.7, 0.3};
auto probs = trainer.predict(sample);
std::cout << "Вероятность класса 1: " << probs[0] << "\n";
```

### predict_class() — предсказание класса (бинарный случай)

```cpp
int predict_class(const std::vector<double>& input) const;
```
- Возвращает `0` или `1` в зависимости от порога `0.5`.
- Удобно для быстрой классификации.

### predict_batch() — пакетное предсказание

```cpp
Matrix<double> predict_batch(const Matrix<double>& inputs) const;
```
- Обрабатывает несколько примеров за один вызов.
- Возвращает столбец предсказаний `[n_samples × 1]`.

## УПРАВЛЕНИЕ КОНФИГУРАЦИЕЙ

```cpp
// Получить текущие настройки
const TrainingConfig& cfg = trainer.getConfig();

// Изменить настройки 
TrainingConfig new_cfg = cfg;
new_cfg.learning_rate = 0.01;
trainer.setConfig(new_cfg);
```

## ОБРАБОТКА ОШИБОК

Библиотека выбрасывает `std::invalid_argument` при:

| Сценарий | Сообщение | Решение |
| :--- | :--- | :--- |
| Несоответствие размеров входов/целей | `train: number of samples mismatch` | Проверить `inputs.rows() == targets.rows()` |
| Ошибка размерностей в слоях | `Forward pass layer X: current rows != weights cols` | Проверить архитектуру сети и входные данные |
| Пустые данные | `train: inputs or targets is empty` | Убедиться, что данные загружены корректно |

**Пример обработки:**
```cpp
try {
    trainer.train(data);
} catch (const std::invalid_argument& e) {
    std::cerr << "Training error: " << e.what() << std::endl;
    // Логика восстановления
}
```

## ПОЛНЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ

```cpp
#include "../Trainer_class/trainer.h"
#include "../CSVParser/pars.h"
#include <iostream>

int main() {
    try {
        // 1. Загрузка данных
        CSVParser parser(',', true);
        auto data = parser.loadTrainingDataAuto("dataset.csv");

        // 2. Создание сети: 2 входа → 4 нейрона (ReLU) → 1 выход (Sigmoid)
        NeuralNetwork net(2, {4, 1}, {Activation::RELU, Activation::SIGMOID});

        // 3. Настройка обучения
        TrainingConfig cfg{.epochs = 1000, .learning_rate = 0.05, .verbose = true};
        Trainer trainer(net, cfg);

        // 4. Обучение
        trainer.train(data);

        // 5. Оценка
        double acc = trainer.evaluate(data);
        std::cout << "Итоговая точность: " << acc << "%\n";

        // 6. Предсказание
        std::vector<double> sample = {0.8, 0.2};
        int cls = trainer.predict_class(sample);
        std::cout << "Предсказанный класс: " << cls << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```
