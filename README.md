# cpp_neural_clustering
Лабораторные работы 1 и 2  по C++

# Работа парсера
### Matrix<double> - при вызове loadToMatrix()

```
Matrix<double> data = parser.loadToMatrix("data.csv");

Получаете матрицу (двумерный массив) чисел типа double, где:

- Строки - соответствуют строкам CSV (без заголовка, если он был)

- Колонки - соответствуют колонкам CSV

- Все данные преобразованы в числа
```

###  Dataset<double> - при вызове loadTrainingData() или loadTrainingDataAuto()

```
Dataset<double> dataset = parser.loadTrainingData("data.csv", input_cols, target_cols);
Получаете структуру, которая содержит:
```

- inputs (Matrix<double>) - матрица входных признаков (features)

- targets (Matrix<double>) - матрица целевых переменных (labels)

- headers (std::vector<std::string>) - заголовки колонок (если has_header = true)

### Matrix<double> - при вызове loadInputsOnly()
```
Matrix<double> inputs = parser.loadInputsOnly("data.csv", {0, 1, 2});
```
Получаете матрицу только с указанными колонками (без целевых переменных).


