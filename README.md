# 📚 Инструкция по работе с библиотекой CSVParser

## Подключение

```cpp
#include "pars.h"
```

## Структуры данных

### Dataset<T>
```cpp
template<typename T>
struct Dataset {
    Matrix<T> inputs;           // Входные признаки
    Matrix<T> targets;          // Целевые переменные
    std::vector<std::string> headers;  // Заголовки колонок
};
```

### TrainTestSplit
```cpp
struct TrainTestSplit {
    Matrix<double> X_train;     // Обучающие признаки
    Matrix<double> X_test;      // Тестовые признаки
    Matrix<double> y_train;     // Обучающие цели
    Matrix<double> y_test;      // Тестовые цели
};
```

## Инициализация

```cpp
// С разделителем-запятой и заголовком
CSVParser parser(',', true);

// С разделителем-точкой с запятой без заголовка
CSVParser parser2(';', false);

// По умолчанию: разделитель ',', есть заголовок
CSVParser parser3;
```

## Основные методы

### 1. `loadClassification2D()` - для 2D классификации
```cpp
Dataset<double> loadClassification2D(const std::string& filename) const;
```
**Формат CSV:** `x, y, class`
- Автоматически определяет колонки (0=x, 1=y, 2=class)
- Выводит статистику датасета

### 2. `loadTrainingData()` - с указанием колонок
```cpp
Dataset<double> loadTrainingData(
    const std::string& filename,
    const std::vector<int>& input_columns,
    const std::vector<int>& target_columns
) const;
```

### 3. `loadTrainingDataAuto()` - автоопределение цели
```cpp
Dataset<double> loadTrainingDataAuto(const std::string& filename) const;
```
- Последняя колонка становится целевой

### 4. `loadInputsOnly()` - только входные данные
```cpp
Matrix<double> loadInputsOnly(
    const std::string& filename,
    const std::vector<int>& input_columns
) const;
```

### 5. `splitTrainTest()` - разделение на train/test
```cpp
TrainTestSplit splitTrainTest(
    const Matrix<double>& X,
    const Matrix<double>& y,
    double test_ratio = 0.2,
    bool shuffle = true
) const;
```

### 6. `getHeaders()` - получение заголовков
```cpp
std::vector<std::string> getHeaders(const std::string& filename) const;
```

### 7. `loadToMatrix()` - загрузка в матрицу
```cpp
Matrix<double> loadToMatrix(const std::string& filename) const;
```

## Обработка ошибок

Библиотека выбрасывает исключения `std::runtime_error` в следующих случаях:

```cpp
try {
    auto data = parser.loadClassification2D("nonexistent.csv");
} catch (const std::runtime_error& e) {
    // Файл не найден
    std::cerr << "File error: " << e.what() << std::endl;
}

try {
    auto data = parser.loadClassification2D("wrong_format.csv");
} catch (const std::runtime_error& e) {
    // Неверный формат (меньше 3 колонок)
    std::cerr << "Format error: " << e.what() << std::endl;
}

try {
    std::vector<int> inputs = {0, 1, 10};  // колонка 10 не существует
    auto data = parser.loadTrainingData("data.csv", inputs, {2});
} catch (const std::runtime_error& e) {
    // Выход за границы колонок
    std::cerr << "Column error: " << e.what() << std::endl;
}
```

