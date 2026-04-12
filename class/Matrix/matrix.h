#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

#ifdef _OPENMP
#include <omp.h>
#endif

// КОНЦЕПТЫ
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept Multipliable = requires(T a, T b) {
    { a * b } -> std::convertible_to<T>;
};

template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

template<typename T>
concept MatrixElement = Arithmetic<T> && Multipliable<T> && Addable<T>;

template<MatrixElement T = double>
class Matrix {
private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

    void multiplyBlock(const Matrix& other, Matrix& result,
                      size_t start_i, size_t start_j, size_t start_k,
                      size_t block_size) const;

public:
    // КОНСТРУКТОРЫ
    Matrix();
    explicit Matrix(size_t rows, size_t cols, T init_value = T{});

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix(std::initializer_list<std::initializer_list<U>> list);

    // Rule of five
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept = default;
    ~Matrix() = default;

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& operator=(std::initializer_list<std::initializer_list<U>> list);

    // ГЕТТЕРЫ
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    // ДОСТУП К ЭЛЕМЕНТАМ
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;

    // МЕТОДЫ ДЛЯ ЗАПОЛНЕНИЯ
    Matrix& set(size_t i, size_t j, T value);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setRow(size_t row, const std::vector<U>& values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setRow(size_t row, std::initializer_list<U> values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setCol(size_t col, const std::vector<U>& values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setCol(size_t col, std::initializer_list<U> values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setAll(const std::vector<U>& values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setAll(std::initializer_list<U> values);

    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setDiagonal(const std::vector<U>& values);

    // Перегрузка для initializer_list
    template<typename U>
    requires std::convertible_to<U, T>
    Matrix& setDiagonal(std::initializer_list<U> values) {
        return setDiagonal(std::vector<U>(values));
    }

    Matrix& setDiagonal(T value);

    // БАЗОВЫЕ ОПЕРАЦИИ
    void fill(T value);
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(const Matrix& other);

    // для тестов
    static Matrix identity(size_t n);
    static Matrix random(size_t rows, size_t cols, T min = T{-1}, T max = T{1});

    // функции доступа к privat переменным
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& mat);

    template<typename U>
    friend Matrix<U> multiplyWithTranspose(const Matrix<U>& a, const Matrix<U>& b);
};

// СВОБОДНЫЕ ФУНКЦИИ
template<MatrixElement T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat);

template<MatrixElement T>
Matrix<T> multiplyWithTranspose(const Matrix<T>& a, const Matrix<T>& b);

// ПОДКЛЮЧЕНИЕ РЕАЛИЗАЦИИ
#include "matrix.cpp"

#endif // MATRIX_H