## Документация класса `NeuralNetwork`

### Общее описание

`NeuralNetwork` - реализация полносвязной нейронной сети с поддержкой:
- произвольного количества слоёв
- нескольких функций активации (Sigmoid, ReLU, Linear, Softmax)
- прямого и обратного распространения ошибки (backpropagation)
- пакетной обработки (`forwardBatch`)
- сохранения/загрузки модели в файл
- логирования

Класс использует собственную библиотеку `Matrix<T>` для матричных операций и структуру `Datasetpars<double>` для работы с наборами данных.

---
### Зависимости

- `Matrix.h` – класс матриц с поддержкой операций умножения, транспонирования и доступа по индексам.
- `pars.h` – определяет структуру `Datasetpars<double>` с полями `inputs` (матрица признаков) и `targets` (матрица целевых значений).

---
## Перечисление `Activation`

Используется для задания функции активации слоя.

| Значение  | Описание                                  |
| --------- | ----------------------------------------- |
| `SIGMOID` | Сигмоида: \( f(x) = \frac{1}{1+e^{-x}} \) |
| `RELU`    | ReLU: \( f(x) = \max(0, x) \)             |
| `LINEAR`  | Линейная: \( f(x) = x \)                  |
| `SOFTMAX` | Softmax (только для выходного слоя)       |

---
## Структура `Layer`

Хранит параметры и промежуточные данные одного слоя.

| Поле | Тип | Описание |
|------|-----|----------|
| `weights` | `Matrix<double>` | Матрица весов (размер `output_size × input_size`) |
| `biases`  | `Matrix<double>` | Вектор смещений (размер `output_size × 1`) |
| `activation` | `Activation` | Функция активации слоя |
| `z` | `Matrix<double>` | Взвешенная сумма до активации (кэш для backward) |
| `a` | `Matrix<double>` | Выход после активации (кэш) |

---

## Класс `NeuralNetwork`

### Конструкторы и деструктор

```cpp
NeuralNetwork(const std::vector<int>& sizes,
              Activation hiddenActivation = Activation::SIGMOID,
              bool enableLogging = false,
              const std::string& logFilename = "");
```

**Параметры:**
- `sizes` - вектор размеров слоёв. Длина не менее 2.
- `hiddenActivation` - активация для всех скрытых слоёв. Выходной слой всегда получает `SIGMOID` (если не изменять вручную через `addLayer` или конструктор - в конструкторе выходной слой принудительно `SIGMOID`).
- `enableLogging` - включить запись логов в файл.
- `logFilename` - имя файла лога (если `enableLogging = true`).

**Исключения:** `std::invalid_argument`, если `sizes.size() < 2`.

```cpp
~NeuralNetwork();
```
Закрывает файл лога, если он был открыт.

### Методы управления архитектурой

```cpp
void addLayer(int outputSize, Activation activation = Activation::SIGMOID);
```
Добавляет новый полносвязный слой в конец сети.
- `outputSize` - количество нейронов в новом слое.
- `activation` - функция активации нового слоя.
- Требует, чтобы сеть уже имела хотя бы один слой.

```cpp
size_t numLayers() const noexcept;
size_t inputSize() const noexcept;
size_t outputSize() const noexcept;
```
Возвращают количество слоёв, размер входа (число признаков) и выхода (число нейронов последнего слоя).

```cpp
std::vector<Layer>& getLayers();
const std::vector<Layer>& getLayers() const;
```
Прямой доступ к слоям.

### Прямой проход (forward)

```cpp
std::vector<double> forward(const std::vector<double>& input);
```
Выполняет прямой проход для одного входного вектора.
- `input` - вектор признаков (должен совпадать с `inputSize()`).
- Возвращает выход последнего слоя (после активации).

```cpp
Matrix<double> forwardBatch(const Matrix<double>& X);
```
Пакетный прямой проход.
- `X` - матрица признаков размером `(inputSize × batchSize)`.
- Возвращает матрицу выходов размером `(outputSize × batchSize)`.

### Обратное распространение и обучение

```cpp
void backward(const std::vector<double>& x,
              const std::vector<double>& y_true,
              std::vector<Matrix<double>>& dW,
              std::vector<Matrix<double>>& db);
```
Вычисляет градиенты для одного примера.
- `x` - входной вектор.
- `y_true` - целевой вектор (one‑hot или скаляр для бинарной классификации).
- `dW`, `db` - выходные векторы градиентов весов и смещений (будут изменены).
- Поддерживаются выходные активации `SIGMOID` (бинарная классификация) и `SOFTMAX` (многоклассовая). Для `SIGMOID` используется MSE‑подобная производная (`a - y_true`).

```cpp
void updateWeights(const std::vector<Matrix<double>>& dW,
                   const std::vector<Matrix<double>>& db,
                   double learningRate);
```
Обновляет веса и смещения по градиентам.
- `dW`, `db` - градиенты, полученные из `backward`.
- `learningRate` - скорость обучения.

### Предсказания

```cpp
double predictProba(const std::vector<double>& input);
```
Для бинарной классификации возвращает вероятность принадлежности к классу 1. Для многоклассовой - возвращает вероятность класса 1.

```cpp
std::vector<double> predictProbabilities(const std::vector<double>& input);
```
Возвращает вероятности для всех классов.
- Если выходной слой имеет 1 нейрон (бинарный выход), результат будет `{1-p, p}`.
- Иначе возвращает выход сети как есть (ожидается, что последний слой имеет Softmax или Sigmoid).

```cpp
int predict(const std::vector<double>& input);
```
Возвращает предсказанный класс:
- Для 2 классов: `1`, если вероятность ≥ 0.5, иначе `0`.
- Для большего числа классов: argmax вероятностей.

```cpp
double accuracy(const Datasetpars<double>& data);
```
Вычисляет долю правильных ответов на наборе данных.
- `data.inputs` - матрица признаков (каждая строка – пример).
- `data.targets` - матрица целевых значений (первый столбец, целые числа).

### Сохранение / загрузка модели

```cpp
bool saveModel(const std::string& filename);
bool loadModel(const std::string& filename);
```
Сохраняют/загружают веса, смещения и архитектуру (функции активации) в текстовом формате.  
**Формат файла:**
```
<num_layers>
<rows> <cols>
<weights_matrix построчно>
<biases_vector>
<activation_int>
...
```
Возвращают `true` при успехе, иначе `false` (и пишут в лог).

### Копирование весов

```cpp
void copyWeightsFrom(const NeuralNetwork& other);
```
Копирует веса и смещения из другой сети. Сети должны иметь идентичную архитектуру (количество и размеры слоёв). Иначе генерируется `std::runtime_error`.

### Вспомогательные статические методы

```cpp
static double sigmoid(double x);
static double sigmoidDerivative(double x);
static double relu(double x);
static double reluDerivative(double x);
static double linear(double x);
static double linearDerivative(double x);
```
Чистые математические функции активации и их производные.

```cpp
static void softmax(Matrix<double>& mat);
```
Применяет softmax к каждому столбцу матрицы (построчно нормализуя).

```cpp
void applyActivation(Matrix<double>& mat, Activation act);
void applyActivationDerivative(Matrix<double>& mat, Activation act);
```
Применяют функцию активации или её производную поэлементно к матрице. Для `SOFTMAX` производная не реализована (выбросит исключение).

### Логирование

```cpp
void printLayers();
```
Выводит в лог/консоль структуру сети (размеры и типы активаций).

Логирование управляется параметрами конструктора. Все сообщения дублируются в консоль, если `toConsole = true` (по умолчанию).

---

## Пример использования (бинарная классификация)

```cpp
#include "neural_net.h"
#include <iostream>

int main() {
    // Создаём сеть: 2 входа, 3 скрытых нейрона (ReLU), 1 выход (Sigmoid)
    std::vector<int> sizes = {2, 3, 1};
    NeuralNetwork nn(sizes, Activation::RELU, true, "log.txt");

    // Обучаем на одном примере (XOR)
    std::vector<double> x = {0, 1};
    std::vector<double> y = {1.0};

    std::vector<Matrix<double>> dW, db;
    nn.backward(x, y, dW, db);
    nn.updateWeights(dW, db, 0.1);

    // Предсказание
    double prob = nn.predictProba(x);
    int pred = nn.predict(x);
    std::cout << "Predicted probability: " << prob << ", class: " << pred << std::endl;

    // Сохранение
    nn.saveModel("model.txt");
    return 0;
}
```

---

## Совместимость

Код написан под C++11 и выше. Для компиляции необходимы:
- `#include <vector>`, `<string>`, `<fstream>`, `<iostream>`, `<random>`, `<stdexcept>`, `<cmath>`, `<algorithm>`, `<numeric>`.
- Собственные модули `Matrix` и `pars`.
