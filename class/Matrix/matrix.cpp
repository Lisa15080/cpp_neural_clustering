#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"

//  КОНСТРУКТОРЫ

template<typename T>
Matrix<T>::Matrix() : rows_(0), cols_(0) {}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T init_value)
    : rows_(rows), cols_(cols), data_(rows * cols, init_value) {}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) {
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
        for (const T& val : row) {
            (*this)(i, j++) = val;
        }
        ++i;
    }
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(std::initializer_list<std::initializer_list<T>> list) {
    *this = Matrix<T>(list);
    return *this;
}

//ДОСТУП К ЭЛЕМЕНТАМ

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

// МЕТОДЫ ЗАПОЛНЕНИЯ
template<typename T>
Matrix<T>& Matrix<T>::set(size_t i, size_t j, T value) {
    (*this)(i, j) = value;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::setRow(size_t row, const std::vector<T>& values) {
    if (values.size() != cols_) {
        throw std::invalid_argument("Row size mismatch: expected " +
                                  std::to_string(cols_) + ", got " +
                                  std::to_string(values.size()));
    }
    for (size_t j = 0; j < cols_; ++j) {
        (*this)(row, j) = values[j];
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::setRow(size_t row, std::initializer_list<T> values) {
    return setRow(row, std::vector<T>(values));
}

template<typename T>
Matrix<T>& Matrix<T>::setCol(size_t col, const std::vector<T>& values) {
    if (values.size() != rows_) {
        throw std::invalid_argument("Column size mismatch: expected " +
                                  std::to_string(rows_) + ", got " +
                                  std::to_string(values.size()));
    }
    for (size_t i = 0; i < rows_; ++i) {
        (*this)(i, col) = values[i];
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::setCol(size_t col, std::initializer_list<T> values) {
    return setCol(col, std::vector<T>(values));
}

template<typename T>
Matrix<T>& Matrix<T>::setAll(const std::vector<T>& values) {
    if (values.size() != rows_ * cols_) {
        throw std::invalid_argument("Values size mismatch");
    }
    data_ = values;
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::setAll(std::initializer_list<T> values) {
    return setAll(std::vector<T>(values));
}

template<typename T>
Matrix<T>& Matrix<T>::setDiagonal(const std::vector<T>& values) {
    size_t n = std::min({rows_, cols_, values.size()});
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = values[i];
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::setDiagonal(T value) {
    size_t n = std::min(rows_, cols_);
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = value;
    }
    return *this;
}

// БАЗОВЫЕ ОПЕРАЦИИ

template<typename T>
void Matrix<T>::fill(T value) {
    std::fill(data_.begin(), data_.end(), value);
}

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

// УМНОЖЕНИЕ
//Вспомогательный метод
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

template<typename T>
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

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    return multiply(other);
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& other) {
    *this = multiply(other);
    return *this;
}

//  СТАТИЧЕСКИЕ МЕТОДЫ

template<typename T>
Matrix<T> Matrix<T>::identity(size_t n) {
    Matrix<T> result(n, n, T{});
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

// для тестов
namespace {
    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, void>::type
    fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<T> dis(min, max);

        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }

    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value, void>::type
    fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);

        for (size_t i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }

    template<typename T>
    typename std::enable_if<!std::is_integral<T>::value && !std::is_floating_point<T>::value, void>::type
    fill_random_impl(std::vector<T>& data, size_t size, T min, T max) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(static_cast<int>(min), static_cast<int>(max));

        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T min, T max) {
    Matrix<T> result(rows, cols);
    fill_random_impl(result.data_, rows * cols, min, max);
    return result;
}

//СВОБОДНЫЕ ФУНКЦИИ
//оператор вывода
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.rows_; ++i) {
        os << "[";
        for (size_t j = 0; j < mat.cols_; ++j) {
            os << std::setw(10) << std::setprecision(4) << mat(i, j);
            if (j < mat.cols_ - 1) os << ", ";
        }
        os << "]\n";
    }
    return os;
}
//для матрицы грамма, вычисления коварционной матрицы
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