// pars.cpp - реализация
#include "pars.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>

// Конструктор
CSVParser::CSVParser(char delimiter, bool has_header)
    : delimiter_(delimiter), has_header_(has_header) {}

std::string CSVParser::cleanToken(const std::string& token) const {
    if (token.empty()) return token;
    
    std::string cleaned = token;

    // 1. Удаляем пробелы, табы, переносы строк
    size_t start = cleaned.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = cleaned.find_last_not_of(" \t\r\n");
    cleaned = cleaned.substr(start, end - start + 1);
    
    if (cleaned.empty()) return cleaned;

    // 2. Удаляем кавычки с обоих концов (повторяем пока есть)
    bool changed = true;
    while (changed && cleaned.size() >= 1) {
        changed = false;
        if (cleaned.front() == '"' || cleaned.front() == '\'') {
            cleaned.erase(0, 1);
            changed = true;
        }
        if (!cleaned.empty() && (cleaned.back() == '"' || cleaned.back() == '\'')) {
            cleaned.pop_back();
            changed = true;
        }
    }

    // 3. Финальная обрезка пробелов
    start = cleaned.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        end = cleaned.find_last_not_of(" \t\r\n");
        cleaned = cleaned.substr(start, end - start + 1);
    }

    return cleaned;
}
// Разбиение строки на токены
std::vector<std::string> CSVParser::splitLine(const std::string& line) const {
        std::vector<std::string> tokens;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, delimiter_)) {
            token = cleanToken(token);  // используем cleanToken
            tokens.push_back(token);
        }

        return tokens;  // ← ВАЖНО: добавить return!
    }


// Преобразование строки в double
double CSVParser::parseDouble(const std::string& token, size_t row, size_t col) const {
    try {
        return std::stod(token);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(
            "CSV parsing error at row " + std::to_string(row) + 
            ", column " + std::to_string(col) + ": '" + token + 
            "' is not a valid number"
        );
    } catch (const std::out_of_range& e) {
        throw std::runtime_error(
            "CSV parsing error at row " + std::to_string(row) + 
            ", column " + std::to_string(col) + ": '" + token + 
            "' is out of range for double"
        );
    }
}

// Проверка валидности колонок
void CSVParser::validateColumns(const Matrix<double>& matrix, 
                                 const std::vector<int>& columns, 
                                 const std::string& context) const {
    size_t max_col = matrix.cols();
    for (int col : columns) {
        if (col < 0 || static_cast<size_t>(col) >= max_col) {
            throw std::runtime_error(
                context + " column index " + std::to_string(col) + 
                " is out of range (max: " + std::to_string(max_col - 1) + ")"
            );
        }
    }
}

// Извлечение колонок
Matrix<double> CSVParser::extractColumns(const Matrix<double>& source, 
                                          const std::vector<int>& columns) const {
    if (columns.empty()) {
        return Matrix<double>();
    }
    
    Matrix<double> result(source.rows(), columns.size());
    
    for (size_t i = 0; i < source.rows(); ++i) {
        for (size_t j = 0; j < columns.size(); ++j) {
            result(i, j) = source(i, columns[j]);
        }
    }
    
    return result;
}
// Получение заголовков колонок
std::vector<std::string> CSVParser::getHeaders(const std::string& filename) const {
    std::vector<std::string> headers;

    if (!has_header_) {
        return headers;  // Возвращаем пустой вектор, если заголовков нет
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    if (std::getline(file, line)) {
        headers = splitLine(line);
    }

    file.close();
    return headers;
}
// Загрузка всего CSV в матрицу
Matrix<double> CSVParser::loadToMatrix(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    size_t expected_cols = 0;
    size_t row_count = 0;
    
    // Пропускаем заголовок
    if (has_header_) {
        std::getline(file, line);
    }
    
    // Читаем данные
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        auto tokens = splitLine(line);
        
        if (expected_cols == 0) {
            expected_cols = tokens.size();
        } else if (tokens.size() != expected_cols) {
            throw std::runtime_error(
                "CSV error at row " + std::to_string(row_count + 1) + 
                ": expected " + std::to_string(expected_cols) + 
                " columns, got " + std::to_string(tokens.size())
            );
        }
        
        std::vector<double> row;
        for (size_t j = 0; j < tokens.size(); ++j) {
            row.push_back(parseDouble(tokens[j], row_count + 1, j + 1));
        }
        
        data.push_back(row);
        row_count++;
    }
    
    file.close();
    
    if (data.empty()) {
        return Matrix<double>();
    }
    
    // Преобразуем в Matrix
    Matrix<double> result(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            result(i, j) = data[i][j];
        }
    }
    
    return result;
}

// СПЕЦИАЛИЗИРОВАННЫЙ МЕТОД ДЛЯ 2D КЛАССИФИКАЦИИ
// Предполагает формат: x, y, class
Datasetpars<double> CSVParser::loadClassification2D(const std::string& filename) const {
    Matrix<double> all_data = loadToMatrix(filename);
    
    if (all_data.rows() == 0) {
        throw std::runtime_error("File is empty: " + filename);
    }
    
    if (all_data.cols() < 3) {
        throw std::runtime_error(
            "Expected at least 3 columns (x, y, class), but got " + 
            std::to_string(all_data.cols())
        );
    }
    
    // Предполагаем: колонка 0 = x, колонка 1 = y, колонка 2 = class
    std::vector<int> input_cols = {0, 1};
    std::vector<int> target_cols = {2};
    
    // Проверяем, что метки классов корректны
    for (size_t i = 0; i < all_data.rows(); ++i) {
        double class_label = all_data(i, 2);
        if (class_label < 0) {
            throw std::runtime_error(
                "Invalid class label at row " + std::to_string(i) + 
                ": " + std::to_string(class_label) + " (should be >= 0)"
            );
        }
    }
    
    Datasetpars<double> result;
    result.inputs = extractColumns(all_data, input_cols);
    result.targets = extractColumns(all_data, target_cols);
    
    if (has_header_) {
        result.headers = getHeaders(filename);
    } else {
        result.headers = {"x", "y", "class"};
    }
    
    // Выводим информацию о датасете
    std::cout << "=== Dataset Info ===" << std::endl;
    std::cout << "Samples: " << result.inputs.rows() << std::endl;
    std::cout << "Features: " << result.inputs.cols() << " (x, y)" << std::endl;
    
    // Подсчитываем уникальные классы
    std::set<double> unique_classes;
    for (size_t i = 0; i < result.targets.rows(); ++i) {
        unique_classes.insert(result.targets(i, 0));
    }
    std::cout << "Classes: " << unique_classes.size() << " (";
    for (auto it = unique_classes.begin(); it != unique_classes.end(); ++it) {
        if (it != unique_classes.begin()) std::cout << ", ";
        std::cout << *it;
    }
    std::cout << ")" << std::endl;
    std::cout << "====================" << std::endl;
    
    return result;
}

Datasetpars<double> CSVParser::loadTrainingData(
    const std::string& filename,
    const std::vector<int>& input_columns,
    const std::vector<int>& target_columns
) const {
    Matrix<double> all_data = loadToMatrix(filename);

    if (all_data.rows() == 0) {
        return Datasetpars<double>();
    }

    validateColumns(all_data, input_columns, "Input");
    validateColumns(all_data, target_columns, "Target");

    Datasetpars<double> result;
    result.inputs = extractColumns(all_data, input_columns);
    result.targets = extractColumns(all_data, target_columns);

    if (has_header_) {
        result.headers = getHeaders(filename);
    }

    return result;
}

// loadTrainingDataAuto метод (автоматическое определение: последняя колонка - цель)
Datasetpars<double> CSVParser::loadTrainingDataAuto(const std::string& filename) const {
    Matrix<double> all_data = loadToMatrix(filename);

    if (all_data.rows() == 0 || all_data.cols() < 2) {
        throw std::runtime_error(
            "Not enough columns for auto-detection. Need at least 2 columns."
        );
    }

    std::vector<int> input_columns;
    for (size_t i = 0; i < all_data.cols() - 1; ++i) {
        input_columns.push_back(static_cast<int>(i));
    }

    std::vector<int> target_columns = {static_cast<int>(all_data.cols() - 1)};

    return loadTrainingData(filename, input_columns, target_columns);
}

// loadInputsOnly метод (загрузка только входных данных)
Matrix<double> CSVParser::loadInputsOnly(
    const std::string& filename,
    const std::vector<int>& input_columns
) const {
    Matrix<double> all_data = loadToMatrix(filename);

    if (all_data.rows() == 0 || input_columns.empty()) {
        return Matrix<double>();
    }

    validateColumns(all_data, input_columns, "Input");
    return extractColumns(all_data, input_columns);
}

