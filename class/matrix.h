#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include <iostream>
#include <iomanip>

template<typename T = double>
class Matrix {
private:
    std::vector<T> data_;
    size_t rows_;
    size_t cols_;

    void multiplyBlock(const Matrix& other, Matrix& result,
                      size_t start_i, size_t start_j, size_t start_k,
                      size_t block_size) const;

public:
    // Конструкторы
    Matrix();
    Matrix(size_t rows, size_t cols, T init_value = T{});

    // Rule of five
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) noexcept = default;
    ~Matrix() = default;

    // Getters
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    // Доступ к элементам
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;

    // Операции
    void fill(T value);
    Matrix transpose() const;
    Matrix multiply(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator*=(const Matrix& other);

    // Статические методы
    static Matrix identity(size_t n);
    static Matrix random(size_t rows, size_t cols, T min = T{-1}, T max = T{1});

    // Дружественная функция для вывода
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& mat);
};

// Свободные функции
template<typename T>
Matrix<T> multiplyWithTranspose(const Matrix<T>& a, const Matrix<T>& b);

#include "matrix.cpp"

#endif // MATRIX_H