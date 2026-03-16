cmake_minimum_required(VERSION 3.14)
project(neuro_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Политика для скачивания (убирает предупреждение)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Google Test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

enable_testing()

# нейросеть + матрицы
add_library(neuro_lib STATIC
    neuro_model/Neural_Net/neural_net.cpp   # путь к файлу нейросети
    class/Matrix/matrix.cpp                  # путь к файлу матриц
)

# Добавляем пути к заголовочным файлам, чтобы компилятор их нашёл
target_include_directories(neuro_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
    ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
)

# Тесты для нейросети
add_executable(neural_tests
    test/test_neural_net.cpp
)

target_link_libraries(neural_tests
    PRIVATE
    neuro_lib
    GTest::gtest_main
)

target_include_directories(neural_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
    ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
)

# Автоматическое обнаружение тестов Google Test
include(GoogleTest)
gtest_discover_tests(neural_tests)
