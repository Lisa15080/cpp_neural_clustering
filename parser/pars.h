#ifndef PARS_H
#define PARS_H

#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include "../class/Matrix/matrix.h"  // Подключаем ваш класс Matrix

// Структура для хранения загруженных данных
template<typename T = double>
struct Dataset {
    Matrix<T> inputs;      // Входные данные (признаки)
    Matrix<T> targets;     // Целевые значения (метки)
    std::vector<std::string> headers;  // Заголовки колонок (опционально)
};

// Класс для парсинга CSV файлов
class CSVParser {
private:
    char delimiter_;       // Разделитель (по умолчанию ',')
    bool has_header_;      // Есть ли заголовок в первой строке

    // Вспомогательная функция для разбиения строки
    std::vector<std::string> splitLine(const std::string& line) const;

    // Преобразование строки в число с обработкой ошибок
    double parseDouble(const std::string& token, size_t row, size_t col) const;

public:
    // Конструктор
    explicit CSVParser(char delimiter = ',', bool has_header = true);

    // Загрузка всего CSV файла в матрицу
    Matrix<double> loadToMatrix(const std::string& filename) const;

    // Загрузка данных для обучения (с разделением на inputs и targets)
    Dataset<double> loadTrainingData(
        const std::string& filename,
        const std::vector<int>& input_columns,   // индексы колонок для входов
        const std::vector<int>& target_columns   // индексы колонок для целей
    ) const;

    // Загрузка с автоматическим определением (последняя колонка - цель)
    Dataset<double> loadTrainingDataAuto(const std::string& filename) const;

    // Загрузка только входных данных (для предсказаний)
    Matrix<double> loadInputsOnly(
        const std::string& filename,
        const std::vector<int>& input_columns
    ) const;

    // Получить заголовки колонок (если есть)
    std::vector<std::string> getHeaders(const std::string& filename) const;

    // Установка разделителя
    void setDelimiter(char delimiter) { delimiter_ = delimiter; }

    // Установка наличия заголовка
    void setHasHeader(bool has_header) { has_header_ = has_header; }
};

#endif // CSV_PARSER_H