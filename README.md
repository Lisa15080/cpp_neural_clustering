# 

Реализация полносвязной нейронной сети на C++ без использования сторонних библиотек глубокого обучения. Проект включает кастомный класс для работы с матрицами, парсер CSV, генератор синтетических данных и  тренер.

## Структура проекта
cpp_neural_clustering
├── class
│   └── Matrix # Математическое ядро
│       ├── matrix.cpp
│       └── matrix.h
├── CMakeLists.txt
├── Kegal_detaset # Датасеты с Kaggle
│   ├── circles_detaset.csv
│   ├── test.csv
│   └── train.csv
├── neuro_model
│   ├── DataSet #Генерация данных
│   │   ├── dataset.cpp
│   │   ├── dataset.h
│   │   └── README.md
│   ├── main.cpp
│   ├── Neural_Net #Ядро нейронной сети
│   │   ├── neural_net.cpp
│   │   ├── neural_net.h
│   │   └── README.md
│   ├── readme.md
│   └── Trainer_class # Логика обучения
│       ├── README.md
│       ├── trainer.cpp
│       └── trainer.h
├── parser # Парсер для работы с внешними данными
│   ├── pars.cpp
│   ├── pars.h
│   └── README.md
├── README.md
└── test #  Модульные тесты
    ├── test_matrix.cpp
    ├── test_neural_net.cpp
    └── test_trainer.cpp
