#include "pars.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Конструктор
CSVParser::CSVParser(char delimiter, bool has_header)
    : delimiter_(delimiter), has_header_(has_header) {}

// Разбиение строки на токены
std::vector<std::string> CSVParser::splitLine(const std::string& line) const {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter_)) {
        // Удаляем пробелы в начале и конце
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        
        // Удаляем кавычки, если есть
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
            token = token.substr(1, token.size() - 2);
        }
        
        tokens.push_back(token);
    }
    
    return tokens;
}

// Преобразование строки в double с обработкой ошибок
double CSVParser::parseDouble(const std::string& token, size_t row, size_t col) const {
    try {
        // Пытаемся преобразовать в double
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

// Загрузка всего CSV файла в матрицу
Matrix<double> CSVParser::loadToMatrix(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    size_t expected_cols = 0;
    size_t row_count = 0;
    
    // Пропускаем заголовок, если он есть
    if (has_header_) {
        std::getline(file, line);
    }
    
    // Читаем данные
    while (std::getline(file, line)) {
        if (line.empty()) continue;  // Пропускаем пустые строки
        
        auto tokens = splitLine(line);
        
        // Проверяем количество колонок
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
        return Matrix<double>();  // Возвращаем пустую матрицу
    }
    
    // Преобразуем vector<vector<double>> в Matrix<double>
    Matrix<double> result(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            result(i, j) = data[i][j];
        }
    }
    
    return result;
}

// Загрузка данных для обучения
Dataset<double> CSVParser::loadTrainingData(
    const std::string& filename,
    const std::vector<int>& input_columns,
    const std::vector<int>& target_columns
) const {
    // Загружаем все данные в матрицу
    Matrix<double> all_data = loadToMatrix(filename);
    
    if (all_data.rows() == 0) {
        return Dataset<double>();  // Пустой датасет
    }
    
    // Проверяем индексы колонок
    size_t max_col = all_data.cols();
    for (int col : input_columns) {
        if (col < 0 || static_cast<size_t>(col) >= max_col) {
            throw std::runtime_error(
                "Input column index " + std::to_string(col) + 
                " is out of range (max: " + std::to_string(max_col - 1) + ")"
            );
        }
    }
    for (int col : target_columns) {
        if (col < 0 || static_cast<size_t>(col) >= max_col) {
            throw std::runtime_error(
                "Target column index " + std::to_string(col) + 
                " is out of range (max: " + std::to_string(max_col - 1) + ")"
            );
        }
    }
    
    // Создаем матрицы для inputs и targets
    Matrix<double> inputs(all_data.rows(), input_columns.size());
    Matrix<double> targets(all_data.rows(), target_columns.size());
    
    // Заполняем inputs
    for (size_t i = 0; i < all_data.rows(); ++i) {
        for (size_t j = 0; j < input_columns.size(); ++j) {
            inputs(i, j) = all_data(i, input_columns[j]);
        }
    }
    
    // Заполняем targets
    for (size_t i = 0; i < all_data.rows(); ++i) {
        for (size_t j = 0; j < target_columns.size(); ++j) {
            targets(i, j) = all_data(i, target_columns[j]);
        }
    }
    
    // Получаем заголовки, если есть
    std::vector<std::string> headers;
    if (has_header_) {
        headers = getHeaders(filename);
    }
    
    return {inputs, targets, headers};
}

// Загрузка с автоматическим определением (последняя колонка - цель)
Dataset<double> CSVParser::loadTrainingDataAuto(const std::string& filename) const {
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

// Загрузка только входных данных
Matrix<double> CSVParser::loadInputsOnly(
    const std::string& filename,
    const std::vector<int>& input_columns
) const {
    Matrix<double> all_data = loadToMatrix(filename);
    
    if (all_data.rows() == 0 || input_columns.empty()) {
        return Matrix<double>();
    }
    
    Matrix<double> inputs(all_data.rows(), input_columns.size());
    
    for (size_t i = 0; i < all_data.rows(); ++i) {
        for (size_t j = 0; j < input_columns.size(); ++j) {
            inputs(i, j) = all_data(i, input_columns[j]);
        }
    }
    
    return inputs;
}

// Получить заголовки колонок
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