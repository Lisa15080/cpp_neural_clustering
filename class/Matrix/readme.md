
## Конструкторы

### Конструктор по умолчанию
```cpp
Matrix<T>::Matrix()
```
Создает пустую матрицу размером 0×0.

### Конструктор с размерами
```cpp
Matrix<T>::Matrix(size_t rows, size_t cols, T init_value = T{})
```
Создает матрицу заданного размера и заполняет ее значением `init_value`.

**Параметры:**
- `rows` - количество строк
- `cols` - количество столбцов
- `init_value` - начальное значение (по умолчанию 0)

**Пример:**
```cpp
Matrix<double> mat(3, 3, 0.0);  // матрица 3x3, заполненная нулями
```

### Конструктор из initializer_list
```cpp
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<U>> list)
```
Создает матрицу из вложенного списка инициализации.

**Пример:**
```cpp
Matrix<double> mat = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
};
```

## 🔧 Операторы доступа

### Оператор ()
```cpp
T& operator()(size_t i, size_t j)
const T& operator()(size_t i, size_t j) const
```
Доступ к элементу матрицы по индексам строки и столбца.

**Параметры:**
- `i` - индекс строки (0-based)
- `j` - индекс столбца (0-based)

**Возвращает:** ссылку на элемент

**Исключения:** `std::out_of_range` при выходе за границы

**Пример:**
```cpp
Matrix<double> mat(3, 3);
mat(0, 0) = 5.0;
double val = mat(0, 0);  // val = 5.0
```

### Оператор присваивания
```cpp
Matrix<T>& operator=(std::initializer_list<std::initializer_list<U>> list)
```
Присваивает матрице значения из initializer_list.

## 📝 Методы заполнения

### set()
```cpp
Matrix<T>& set(size_t i, size_t j, T value)
```
Устанавливает значение элемента и возвращает ссылку на матрицу (для цепочечных вызовов).

**Пример:**
```cpp
mat.set(0, 0, 1).set(0, 1, 2).set(1, 0, 3);
```

### setRow()
```cpp
Matrix<T>& setRow(size_t row, const std::vector<U>& values)
Matrix<T>& setRow(size_t row, std::initializer_list<U> values)
```
Устанавливает значения целой строки.

**Пример:**
```cpp
mat.setRow(0, {1, 2, 3});
mat.setRow(1, std::vector<double>{4, 5, 6});
```

### setCol()
```cpp
Matrix<T>& setCol(size_t col, const std::vector<U>& values)
Matrix<T>& setCol(size_t col, std::initializer_list<U> values)
```
Устанавливает значения целого столбца.

### setAll()
```cpp
Matrix<T>& setAll(const std::vector<U>& values)
Matrix<T>& setAll(std::initializer_list<U> values)
```
Заполняет всю матрицу значениями из вектора (построчно).

### setDiagonal()
```cpp
Matrix<T>& setDiagonal(const std::vector<U>& values)
Matrix<T>& setDiagonal(T value)
```
Устанавливает диагональные элементы.

### fill()
```cpp
void fill(T value)
```
Заполняет всю матрицу указанным значением.

##  Базовые операции

### transpose()
```cpp
Matrix<T> transpose() const
```
Возвращает транспонированную матрицу.

**Пример:**
```cpp
Matrix<double> a = {{1, 2}, {3, 4}};
Matrix<double> b = a.transpose();  // {{1, 3}, {2, 4}}
```

##  Умножение матриц

### multiply()
```cpp
Matrix<T> multiply(const Matrix& other) const
```
Выполняет матричное умножение с оптимизациями:
- Использует блочное умножение для больших матриц
- Пропускает нулевые элементы
- Поддерживает OpenMP параллелизацию

### operator*()
```cpp
Matrix<T> operator*(const Matrix& other) const
```
Аналог `multiply()` для удобного синтаксиса.

### operator*=()
```cpp
Matrix<T>& operator*=(const Matrix& other)
```
Умножает текущую матрицу на другую (заменяет содержимое).

**Пример:**
```cpp
Matrix<double> a = {{1, 2}, {3, 4}};
Matrix<double> b = {{5, 6}, {7, 8}};
Matrix<double> c = a * b;  // матричное умножение
a *= b;  // a = a * b
```

## Статические методы

### identity()
```cpp
static Matrix<T> identity(size_t n)
```
Создает единичную матрицу размером n×n.

**Пример:**
```cpp
auto I = Matrix<double>::identity(3);
// I = [[1, 0, 0],
//      [0, 1, 0],
//      [0, 0, 1]]
```

### random()
```cpp
static Matrix<T> random(size_t rows, size_t cols, T min, T max)
```
Создает матрицу со случайными значениями в диапазоне [min, max].

**Пример:**
```cpp
auto rand_mat = Matrix<double>::random(5, 5, -1.0, 1.0);
```

##  Геттеры

```cpp
size_t rows() const noexcept  // количество строк
size_t cols() const noexcept  // количество столбцов
size_t size() const noexcept   // общее количество элементов
bool isEmpty() const noexcept  // проверка на пустоту
```

##  Свободные функции

### operator<<()
```cpp
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat)
```
Выводит матрицу в поток в удобном для чтения формате.

**Пример:**
```cpp
Matrix<double> mat = {{1, 2}, {3, 4}};
std::cout << mat;
// Вывод:
// [         1,          2]
// [         3,          4]
```

### multiplyWithTranspose()
```cpp
Matrix<T> multiplyWithTranspose(const Matrix<T>& a, const Matrix<T>& b)
```
Вычисляет произведение aᵀ × b. Полезна для вычисления ковариационных матриц.

**Пример:**
```cpp
Matrix<double> gram = multiplyWithTranspose(X, X);  // Xᵀ × X
```

## ⚠️ Исключения

Методы могут выбрасывать:
- `std::invalid_argument` - при несоответствии размеров
- `std::out_of_range` - при выходе за границы матрицы


## 📁 Зависимости

```cpp
#include <algorithm>  // std::fill, std::min
#include <random>     // генерация случайных чисел
#include <iomanip>    // форматированный вывод
#include <stdexcept>  // исключения
```
