// pars.h - обновленный заголовочный файл
#ifndef PARS_H
#define PARS_H
#include <set>
#include <vector>
#include <string>
#include "../class/Matrix/matrix.h"

template<typename T>
struct Datasetpars {
    Matrix<T> inputs;    // признаки (x, y координаты)
    Matrix<T> targets;   // метки классов (0, 1, 2, ...)
    std::vector<std::string> headers;
};

class CSVParser {
private:
    char delimiter_;
    bool has_header_;
    
    // Вспомогательные методы
    std::vector<std::string> splitLine(const std::string& line) const;
    double parseDouble(const std::string& token, size_t row, size_t col) const;
    void validateColumns(const Matrix<double>& matrix, 
                        const std::vector<int>& columns, 
                        const std::string& context) const;
    Matrix<double> extractColumns(const Matrix<double>& source, 
                                  const std::vector<int>& columns) const;



public:
    std::string cleanToken(const std::string& token) const;
    // Конструктор
    CSVParser(char delimiter = ',', bool has_header = true);
    
    // Основные методы
    Matrix<double> loadToMatrix(const std::string& filename) const;
    
    // Специализированный метод для 2D классификации
    Datasetpars<double> loadClassification2D(const std::string& filename) const;
    
    // Общие методы загрузки
    Datasetpars<double> loadTrainingData(
        const std::string& filename,
        const std::vector<int>& input_columns,
        const std::vector<int>& target_columns
    ) const;
    
    Datasetpars<double> loadTrainingDataAuto(const std::string& filename) const;
    Matrix<double> loadInputsOnly(const std::string& filename,
                                  const std::vector<int>& input_columns) const;
    
    // Метод для получения заголовков
    std::vector<std::string> getHeaders(const std::string& filename) const;
    
    // Новый метод: разделение на train/test
    struct TrainTestSplit {
        Matrix<double> X_train, X_test, y_train, y_test;
    };
    
    TrainTestSplit splitTrainTest(const Matrix<double>& X,
                                  const Matrix<double>& y,
                                  double test_ratio = 0.2,
                                  bool shuffle = true) const;
};

#endif