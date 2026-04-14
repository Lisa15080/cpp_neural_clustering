cmake_minimum_required(VERSION 3.14)
project(neuro_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Политика для скачивания (убирает предупреждение)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ===== GOOGLE TEST =====
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
#        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
#)
#FetchContent_MakeAvailable(googletest)

#enable_testing()

# ===== ОСНОВНАЯ БИБЛИОТЕКА =====
add_library(neuro_lib STATIC
        neuro_model/Neural_Net/neural_net.cpp
        class/Matrix/matrix.cpp
        parser/pars.cpp
        neuro_model/Trainer_class/trainer.cpp
        neuro_model/DataSet/dataset.cpp
)

target_include_directories(neuro_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
        ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
)
# ===== ИСПОЛНЯЕМЫЙ ФАЙЛ ДЛЯ MAIN =====
add_executable(main neuro_model/main.cpp)  # или ваш файл с main

# Подключаем библиотеку neuro_lib
target_link_libraries(main PRIVATE neuro_lib)

# Добавляем пути для include
target_include_directories(main PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
        ${CMAKE_CURRENT_SOURCE_DIR}/Trainer_class
        ${CMAKE_CURRENT_SOURCE_DIR}/DataSet
        ${CMAKE_CURRENT_SOURCE_DIR}/parser
)
# ===== ТЕСТЫ ДЛЯ НЕЙРОСЕТИ =====


#add_executable(neural_tests
#        test/test_neural_net.cpp
#)

#target_link_libraries(neural_tests
#        PRIVATE
#        neuro_lib
#        GTest::gtest_main
#)

#target_include_directories(neural_tests PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
#        ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
#)

# ===== ТЕСТЫ ДЛЯ МАТРИЦ =====
#add_executable(matrix_tests
#        test/test_matrix.cpp
#        parser/pars.cpp
#        parser/pars.h
#)

#target_link_libraries(matrix_tests
#        PRIVATE
#        neuro_lib
#        GTest::gtest_main
#)

#target_include_directories(matrix_tests PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}
#        ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
#)

# ===== АВТОМАТИЧЕСКОЕ ОБНАРУЖЕНИЕ ТЕСТОВ =====
#include(GoogleTest)
#gtest_discover_tests(neural_tests)
#gtest_discover_tests(matrix_tests)
