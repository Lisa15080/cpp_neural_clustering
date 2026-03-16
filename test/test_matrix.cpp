#include "../class/Matrix/matrix.h"
#include <iostream>

#include <gtest/gtest.h>
#include <type_traits>

// ==================== ТЕСТЫ ДЛЯ РАЗНЫХ ТИПОВ ДАННЫХ ====================

template<typename T>
class MatrixTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Пустая матрица
        empty = Matrix<T>();

        // Матрица 2x2
        m2x2 = Matrix<T>(2, 2);
        m2x2(0, 0) = T{1}; m2x2(0, 1) = T{2};
        m2x2(1, 0) = T{3}; m2x2(1, 1) = T{4};

        // Матрица 2x3
        m2x3 = Matrix<T>(2, 3);
        m2x3(0, 0) = T{1}; m2x3(0, 1) = T{2}; m2x3(0, 2) = T{3};
        m2x3(1, 0) = T{4}; m2x3(1, 1) = T{5}; m2x3(1, 2) = T{6};

        // Матрица 3x2
        m3x2 = Matrix<T>(3, 2);
        m3x2(0, 0) = T{7}; m3x2(0, 1) = T{8};
        m3x2(1, 0) = T{9}; m3x2(1, 1) = T{10};
        m3x2(2, 0) = T{11}; m3x2(2, 1) = T{12};
    }

    Matrix<T> empty;
    Matrix<T> m2x2;
    Matrix<T> m2x3;
    Matrix<T> m3x2;
};

// Тестируем для разных типов
using TestTypes = ::testing::Types<double, float, int>;
TYPED_TEST_SUITE(MatrixTest, TestTypes);

// ==================== ТЕСТЫ КОНСТРУКТОРОВ ====================

TYPED_TEST(MatrixTest, DefaultConstructor) {
    EXPECT_EQ(this->empty.rows(), 0);
    EXPECT_EQ(this->empty.cols(), 0);
    EXPECT_EQ(this->empty.size(), 0);
}

TYPED_TEST(MatrixTest, SizeConstructor) {
    Matrix<TypeParam> m(3, 4, TypeParam{42});

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_EQ(m.size(), 12);

    // Проверяем инициализацию
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(m(i, j), TypeParam{42});
        }
    }
}

TYPED_TEST(MatrixTest, InitializerListConstructor) {
    Matrix<TypeParam> m = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(0, 1), TypeParam{2});
    EXPECT_EQ(m(0, 2), TypeParam{3});
    EXPECT_EQ(m(1, 0), TypeParam{4});
    EXPECT_EQ(m(1, 1), TypeParam{5});
    EXPECT_EQ(m(1, 2), TypeParam{6});
    EXPECT_EQ(m(2, 0), TypeParam{7});
    EXPECT_EQ(m(2, 1), TypeParam{8});
    EXPECT_EQ(m(2, 2), TypeParam{9});
}

TYPED_TEST(MatrixTest, InitializerListConstructorInvalid) {
    // ИСПРАВЛЕНО: EXPECT_THROW с правильным синтаксисом
    EXPECT_THROW(
        Matrix<TypeParam>({
            {1, 2, 3},
            {4, 5},
            {7, 8, 9}
        }),
        std::invalid_argument
    );
}

TYPED_TEST(MatrixTest, AssignmentOperator) {
    Matrix<TypeParam> m;
    m = {
        {1, 2},
        {3, 4}
    };

    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 2);
    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{4});
}

// ==================== ТЕСТЫ ДОСТУПА К ЭЛЕМЕНТАМ ====================

TYPED_TEST(MatrixTest, ElementAccess) {
    auto& m = this->m2x2;

    // Чтение
    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{4});

    // Запись
    m(0, 1) = TypeParam{99};
    EXPECT_EQ(m(0, 1), TypeParam{99});
}

TYPED_TEST(MatrixTest, ElementAccessOutOfRange) {
    auto& m = this->m2x2;

    EXPECT_THROW(m(2, 0), std::out_of_range);
    EXPECT_THROW(m(0, 2), std::out_of_range);
    EXPECT_THROW(m(2, 2), std::out_of_range);
}

TYPED_TEST(MatrixTest, ConstElementAccess) {
    const auto& m = this->m2x2;

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{4});
}

// ==================== ТЕСТЫ МЕТОДОВ FILL ====================

TYPED_TEST(MatrixTest, Fill) {
    Matrix<TypeParam> m(2, 3);
    m.fill(TypeParam{7});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(m(i, j), TypeParam{7});
        }
    }
}

// ==================== ТЕСТЫ МЕТОДОВ SET ====================

TYPED_TEST(MatrixTest, SetMethod) {
    Matrix<TypeParam> m(2, 2);
    m.set(0, 0, TypeParam{1})
     .set(0, 1, TypeParam{2})
     .set(1, 0, TypeParam{3})
     .set(1, 1, TypeParam{4});

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(0, 1), TypeParam{2});
    EXPECT_EQ(m(1, 0), TypeParam{3});
    EXPECT_EQ(m(1, 1), TypeParam{4});
}

TYPED_TEST(MatrixTest, SetRow) {
    Matrix<TypeParam> m(3, 3);
    m.setRow(0, {1, 2, 3})
     .setRow(1, {4, 5, 6})
     .setRow(2, {7, 8, 9});

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{5});
    EXPECT_EQ(m(2, 2), TypeParam{9});
}

TYPED_TEST(MatrixTest, SetRowInvalid) {
    Matrix<TypeParam> m(2, 2);
    EXPECT_THROW(m.setRow(0, {1, 2, 3}), std::invalid_argument);
}

TYPED_TEST(MatrixTest, SetCol) {
    Matrix<TypeParam> m(3, 3);
    m.setCol(0, {1, 2, 3})
     .setCol(1, {4, 5, 6})
     .setCol(2, {7, 8, 9});

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{5});
    EXPECT_EQ(m(2, 2), TypeParam{9});
}

TYPED_TEST(MatrixTest, SetColInvalid) {
    Matrix<TypeParam> m(2, 2);
    EXPECT_THROW(m.setCol(0, {1, 2, 3}), std::invalid_argument);
}

TYPED_TEST(MatrixTest, SetAll) {
    Matrix<TypeParam> m(2, 2);
    m.setAll({1, 2, 3, 4});

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(0, 1), TypeParam{2});
    EXPECT_EQ(m(1, 0), TypeParam{3});
    EXPECT_EQ(m(1, 1), TypeParam{4});
}

TYPED_TEST(MatrixTest, SetAllInvalid) {
    Matrix<TypeParam> m(2, 2);
    EXPECT_THROW(m.setAll({1, 2, 3}), std::invalid_argument);
}

TYPED_TEST(MatrixTest, SetDiagonal) {
    Matrix<TypeParam> m(3, 3);
    m.setDiagonal({1, 2, 3});

    EXPECT_EQ(m(0, 0), TypeParam{1});
    EXPECT_EQ(m(1, 1), TypeParam{2});
    EXPECT_EQ(m(2, 2), TypeParam{3});
    EXPECT_EQ(m(0, 1), TypeParam{0});
}

TYPED_TEST(MatrixTest, SetDiagonalSingle) {
    Matrix<TypeParam> m(3, 3);
    m.setDiagonal(TypeParam{5});

    EXPECT_EQ(m(0, 0), TypeParam{5});
    EXPECT_EQ(m(1, 1), TypeParam{5});
    EXPECT_EQ(m(2, 2), TypeParam{5});
}

// ==================== ТЕСТЫ ТРАНСПОНИРОВАНИЯ ====================

TYPED_TEST(MatrixTest, Transpose) {
    auto t = this->m2x3.transpose();

    EXPECT_EQ(t.rows(), 3);
    EXPECT_EQ(t.cols(), 2);

    EXPECT_EQ(t(0, 0), TypeParam{1});
    EXPECT_EQ(t(0, 1), TypeParam{4});
    EXPECT_EQ(t(1, 0), TypeParam{2});
    EXPECT_EQ(t(1, 1), TypeParam{5});
    EXPECT_EQ(t(2, 0), TypeParam{3});
    EXPECT_EQ(t(2, 1), TypeParam{6});
}

TYPED_TEST(MatrixTest, DoubleTranspose) {
    auto tt = this->m2x3.transpose().transpose();

    EXPECT_EQ(tt.rows(), this->m2x3.rows());
    EXPECT_EQ(tt.cols(), this->m2x3.cols());

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(tt(i, j), this->m2x3(i, j));
        }
    }
}

// ==================== ТЕСТЫ УМНОЖЕНИЯ ====================

TYPED_TEST(MatrixTest, Multiply2x2) {
    Matrix<TypeParam> a = {
        {1, 2},
        {3, 4}
    };

    Matrix<TypeParam> b = {
        {5, 6},
        {7, 8}
    };

    auto result = a * b;

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_EQ(result(0, 0), TypeParam{19});
    EXPECT_EQ(result(0, 1), TypeParam{22});
    EXPECT_EQ(result(1, 0), TypeParam{43});
    EXPECT_EQ(result(1, 1), TypeParam{50});
}

TYPED_TEST(MatrixTest, Multiply2x3With3x2) {
    Matrix<TypeParam> a = {
        {1, 2, 3},
        {4, 5, 6}
    };

    Matrix<TypeParam> b = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    auto result = a * b;

    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 2);
    EXPECT_EQ(result(0, 0), TypeParam{58});
    EXPECT_EQ(result(0, 1), TypeParam{64});
    EXPECT_EQ(result(1, 0), TypeParam{139});
    EXPECT_EQ(result(1, 1), TypeParam{154});
}

TYPED_TEST(MatrixTest, MultiplyInvalidDimensions) {
    Matrix<TypeParam> a(2, 3);
    Matrix<TypeParam> b(2, 2);

    EXPECT_THROW(a * b, std::invalid_argument);
}

TYPED_TEST(MatrixTest, MultiplyAndAssign) {
    Matrix<TypeParam> a = {
        {1, 2},
        {3, 4}
    };

    Matrix<TypeParam> b = {
        {5, 6},
        {7, 8}
    };

    a *= b;

    EXPECT_EQ(a(0, 0), TypeParam{19});
    EXPECT_EQ(a(0, 1), TypeParam{22});
    EXPECT_EQ(a(1, 0), TypeParam{43});
    EXPECT_EQ(a(1, 1), TypeParam{50});
}

// ==================== ТЕСТЫ СТАТИЧЕСКИХ МЕТОДОВ ====================

TYPED_TEST(MatrixTest, Identity) {
    auto I = Matrix<TypeParam>::identity(3);

    EXPECT_EQ(I.rows(), 3);
    EXPECT_EQ(I.cols(), 3);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_EQ(I(i, j), TypeParam{1});
            } else {
                EXPECT_EQ(I(i, j), TypeParam{0});
            }
        }
    }
}

TYPED_TEST(MatrixTest, IdentityMultiplication) {
    auto I = Matrix<TypeParam>::identity(2);
    auto& a = this->m2x2;

    auto result = I * a;

    EXPECT_EQ(result(0, 0), a(0, 0));
    EXPECT_EQ(result(0, 1), a(0, 1));
    EXPECT_EQ(result(1, 0), a(1, 0));
    EXPECT_EQ(result(1, 1), a(1, 1));
}

TYPED_TEST(MatrixTest, Random) {
    auto m = Matrix<TypeParam>::random(10, 10, TypeParam{-5}, TypeParam{5});

    EXPECT_EQ(m.rows(), 10);
    EXPECT_EQ(m.cols(), 10);
}

// ==================== ТЕСТЫ СВОБОДНЫХ ФУНКЦИЙ ====================

TYPED_TEST(MatrixTest, MultiplyWithTranspose) {
    Matrix<TypeParam> a = {
        {1, 2, 3},
        {4, 5, 6}
    };

    Matrix<TypeParam> b = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    auto result = multiplyWithTranspose(a, b);
    auto expected = a * b;

    EXPECT_EQ(result.rows(), expected.rows());
    EXPECT_EQ(result.cols(), expected.cols());
}

// ==================== ТЕСТЫ ГРАНИЧНЫХ СЛУЧАЕВ ====================

TYPED_TEST(MatrixTest, EmptyMatrix) {
    Matrix<TypeParam> empty;

    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);

    auto t = empty.transpose();
    EXPECT_EQ(t.rows(), 0);
    EXPECT_EQ(t.cols(), 0);
}

TYPED_TEST(MatrixTest, ZeroMatrix) {
    Matrix<TypeParam> zero(2, 2, TypeParam{0});
    auto& a = this->m2x2;

    auto result = zero * a;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(result(i, j), TypeParam{0});
        }
    }
}

TYPED_TEST(MatrixTest, OneByOneMatrix) {
    Matrix<TypeParam> m(1, 1);
    m(0, 0) = TypeParam{42};

    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_EQ(m(0, 0), TypeParam{42});

    auto t = m.transpose();
    EXPECT_EQ(t(0, 0), TypeParam{42});
}

// ==================== ТЕСТЫ ДЛЯ INT ====================

TEST(IntegerMatrixTest, IntegerSpecific) {
    Matrix<int> m = {
        {1, 2},
        {3, 4}
    };

    auto result = m * m;

    EXPECT_EQ(result(0, 0), 7);
    EXPECT_EQ(result(0, 1), 10);
    EXPECT_EQ(result(1, 0), 15);
    EXPECT_EQ(result(1, 1), 22);
}

// ==================== ТЕСТЫ ДЛЯ FLOAT ====================

TEST(FloatMatrixTest, FloatPrecision) {
    Matrix<float> a = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    Matrix<float> b = {
        {0.1f, 0.2f},
        {0.3f, 0.4f}
    };

    auto result = a * b;

    EXPECT_NEAR(result(0, 0), 1.0f*0.1f + 2.0f*0.3f, 1e-5);
    EXPECT_NEAR(result(0, 1), 1.0f*0.2f + 2.0f*0.4f, 1e-5);
}

// ==================== MAIN ФУНКЦИЯ (не нужна, GTEST предоставляет свою) ====================}