#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

// Конструкторы
template<typename T>
Matrix<T>::Matrix() : rows_(0), cols_(0) {}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

// Доступ к элементам с проверкой границ
template<typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[i * cols_ + j];
}

template<typename T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[i * cols_ + j];
}

// Заполнение
template<typename T>
void Matrix<T>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Транспонирование
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// Реализация блочного умножения
template<typename T>
void Matrix<T>::multiplyBlock(const Matrix& other, Matrix& result,
                             size_t start_i, size_t start_j, size_t start_k,
                             size_t block_size) const {
    size_t end_i = std::min(start_i + block_size, rows_);
    size_t end_j = std::min(start_j + block_size, other.cols_);
    size_t end_k = std::min(start_k + block_size, cols_);

    for (size_t i = start_i; i < end_i; ++i) {
        for (size_t k = start_k; k < end_k; ++k) {
            T aik = (*this)(i, k);
            if (aik != T{}) {
                for (size_t j = start_j; j < end_j; ++j) {
                    result(i, j) += aik * other(k, j);
                }
            }
        }
    }
}

// Умножение с оптимизациями
template<typename T>
Matrix<T> Matrix<T>::multiply(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }

    Matrix<T> result(rows_, other.cols_, T{});

    // Оптимизация для малых матриц
    if (rows_ * cols_ * other.cols_ < 1024) {
        // Наивное умножение
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                T aik = (*this)(i, k);
                if (aik != T{}) {
                    for (size_t j = 0; j < other.cols_; ++j) {
                        result(i, j) += aik * other(k, j);
                    }
                }
            }
        }
    } else {
        // Блочное умножение
        const size_t BLOCK_SIZE = 64;

        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) if(rows_ * other.cols_ > 10000)
        #endif
        for (size_t i = 0; i < rows_; i += BLOCK_SIZE) {
            for (size_t j = 0; j < other.cols_; j += BLOCK_SIZE) {
                for (size_t k = 0; k < cols_; k += BLOCK_SIZE) {
                    multiplyBlock(other, result, i, j, k, BLOCK_SIZE);
                }
            }
        }
    }

    return result;
}

// Операторы
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    return multiply(other);
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
    *this = multiply(other);
    return *this;
}

// Статические методы
template<typename T>
Matrix<T> Matrix<T>::identity(size_t n) {
    Matrix<T> result(n, n, T{});
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

// Вспомогательные функции для random с использованием SFINAE
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
fill_random(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
fill_random(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

template<typename T>
typename std::enable_if<!std::is_integral<T>::value && !std::is_floating_point<T>::value, void>::type
fill_random(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(static_cast<int>(min), static_cast<int>(max));

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dis(gen));
    }
}

// Метод random с использованием вспомогательных функций
template<typename T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T min, T max) {
    Matrix<T> result(rows, cols);
    fill_random(result.data_, rows * cols, min, max);
    return result;
}

// Оператор вывода
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.rows(); ++i) {
        os << "[";
        for (size_t j = 0; j < mat.cols(); ++j) {
            os << std::setw(10) << std::setprecision(4) << mat(i, j);
            if (j < mat.cols() - 1) os << ", ";
        }
        os << "]\n";
    }
    return os;
}

// Умножение с транспонированием
template<typename T>
Matrix<T> multiplyWithTranspose(const Matrix<T>& a, const Matrix<T>& b) {
    Matrix<T> bt = b.transpose();
    Matrix<T> result(a.rows(), b.cols());

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            T sum = T{};
            for (size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * bt(j, k);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Явная инстанциация для часто используемых типов
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;

// Явная инстанциация для свободных функций
template std::ostream& operator<< <double>(std::ostream&, const Matrix<double>&);
template std::ostream& operator<< <float>(std::ostream&, const Matrix<float>&);
template std::ostream& operator<< <int>(std::ostream&, const Matrix<int>&);

template Matrix<double> multiplyWithTranspose<double>(const Matrix<double>&, const Matrix<double>&);
template Matrix<float> multiplyWithTranspose<float>(const Matrix<float>&, const Matrix<float>&);
template Matrix<int> multiplyWithTranspose<int>(const Matrix<int>&, const Matrix<int>&);

#endif // MATRIX_CPP