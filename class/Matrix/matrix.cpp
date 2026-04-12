#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"

//  КОНСТРУКТОРЫ

template<MatrixElement T>
Matrix<T>::Matrix() : rows_(0), cols_(0) {}

template<MatrixElement T>
Matrix<T>::Matrix(size_t rows, size_t cols, T init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<U>> list) {
    rows_ = list.size();
    if (rows_ == 0) {
        cols_ = 0;
        return;
    }

    cols_ = list.begin()->size();
    data_.resize(rows_ * cols_);

    size_t i = 0;
    for (const auto& row : list) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same size");
        }
        size_t j = 0;
        for (const U& val : row) {
            (*this)(i, j++) = static_cast<T>(val);
        }
        ++i;
    }
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::operator=(std::initializer_list<std::initializer_list<U>> list) {
    *this = Matrix<T>(list);
    return *this;
}

// ДОСТУП К ЭЛЕМЕНТАМ

template<MatrixElement T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[i * cols_ + j];
}

template<MatrixElement T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[i * cols_ + j];
}

// МЕТОДЫ ЗАПОЛНЕНИЯ
template<MatrixElement T>
Matrix<T>& Matrix<T>::set(size_t i, size_t j, T value) {
    (*this)(i, j) = value;
    return *this;
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setRow(size_t row, const std::vector<U>& values) {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of range");
    }
    if (values.size() != cols_) {
        throw std::invalid_argument("Row size mismatch: expected " +
                                  std::to_string(cols_) + ", got " +
                                  std::to_string(values.size()));
    }
    for (size_t j = 0; j < cols_; ++j) {
        (*this)(row, j) = static_cast<T>(values[j]);
    }
    return *this;
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setRow(size_t row, std::initializer_list<U> values) {
    return setRow(row, std::vector<U>(values));
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setCol(size_t col, const std::vector<U>& values) {
    if (col >= cols_) {
        throw std::out_of_range("Column index out of range");
    }
    if (values.size() != rows_) {
        throw std::invalid_argument("Column size mismatch: expected " +
                                  std::to_string(rows_) + ", got " +
                                  std::to_string(values.size()));
    }
    for (size_t i = 0; i < rows_; ++i) {
        (*this)(i, col) = static_cast<T>(values[i]);
    }
    return *this;
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setCol(size_t col, std::initializer_list<U> values) {
    return setCol(col, std::vector<U>(values));
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setAll(const std::vector<U>& values) {
    if (values.size() != rows_ * cols_) {
        throw std::invalid_argument("Values size mismatch: expected " +
                                  std::to_string(rows_ * cols_) + ", got " +
                                  std::to_string(values.size()));
    }
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        data_[i] = static_cast<T>(values[i]);
    }
    return *this;
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setAll(std::initializer_list<U> values) {
    return setAll(std::vector<U>(values));
}

template<MatrixElement T>
template<typename U>
requires std::convertible_to<U, T>
Matrix<T>& Matrix<T>::setDiagonal(const std::vector<U>& values) {
    size_t n = std::min({rows_, cols_, values.size()});
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = static_cast<T>(values[i]);
    }
    return *this;
}

template<MatrixElement T>
Matrix<T>& Matrix<T>::setDiagonal(T value) {
    size_t n = std::min(rows_, cols_);
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = value;
    }
    return *this;
}

// БАЗОВЫЕ ОПЕРАЦИИ

template<MatrixElement T>
void Matrix<T>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

template<MatrixElement T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

// УМНОЖЕНИЕ
// Вспомогательный метод
template<MatrixElement T>
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

template<MatrixElement T>
Matrix<T> Matrix<T>::multiply(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    }

    Matrix<T> result(rows_, other.cols_, T{});

    // Оптимизация для малых матриц
    if (rows_ * cols_ * other.cols_ < 1024) {
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

template<MatrixElement T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    return multiply(other);
}

template<MatrixElement T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
    *this = multiply(other);
    return *this;
}

//  СТАТИЧЕСКИЕ МЕТОДЫ

template<MatrixElement T>
Matrix<T> Matrix<T>::identity(size_t n) {
    Matrix<T> result(n, n, T{});
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

// Вспомогательные функции для random с использованием концептов
template<MatrixElement T>
requires std::is_integral_v<T>
void fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

template<MatrixElement T>
requires std::is_floating_point_v<T>
void fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

template<MatrixElement T>
requires (!std::is_integral_v<T> && !std::is_floating_point_v<T>)
void fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Для нечисловых типов пытаемся преобразовать из int
    std::uniform_int_distribution<int> dis(static_cast<int>(min), static_cast<int>(max));

    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dis(gen));
    }
}

template<MatrixElement T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T min, T max) {
    Matrix<T> result(rows, cols);
    fill_random_impl(result.data_, rows * cols, min, max);
    return result;
}

// СВОБОДНЫЕ ФУНКЦИИ
// оператор вывода
template<MatrixElement T>
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

// для матрицы грамма, вычисления ковариационной матрицы
template<MatrixElement T>
Matrix<T> multiplyWithTranspose(const Matrix<T>& a, const Matrix<T>& b) {
    if (a.cols() != b.cols()) {
        throw std::invalid_argument("Matrices must have same number of columns for AT * B");
    }

    Matrix<T> result(a.rows(), b.rows());

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.rows(); ++j) {
            T sum = T{};
            for (size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(j, k);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// ЯВНАЯ ИНСТАНЦИАЦИЯ
template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;

template std::ostream& operator<< <double>(std::ostream&, const Matrix<double>&);
template std::ostream& operator<< <float>(std::ostream&, const Matrix<float>&);
template std::ostream& operator<< <int>(std::ostream&, const Matrix<int>&);

template Matrix<double> multiplyWithTranspose<double>(const Matrix<double>&, const Matrix<double>&);
template Matrix<float> multiplyWithTranspose<float>(const Matrix<float>&, const Matrix<float>&);
template Matrix<int> multiplyWithTranspose<int>(const Matrix<int>&, const Matrix<int>&);

#endif // MATRIX_CPP