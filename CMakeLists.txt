cmake_minimum_required(VERSION 3.14)

project(neuro_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()


set(PROJECT_SOURCES
        neuro_model/Neural_Net/neural_net.cpp
        neuro_model/Trainer_class/trainer.cpp
        neuro_model/DataSet/dataset.cpp
        parser/pars.cpp
)

set(PROJECT_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/class/Matrix
        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Neural_Net
        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/Trainer_class
        ${CMAKE_CURRENT_SOURCE_DIR}/neuro_model/DataSet
        ${CMAKE_CURRENT_SOURCE_DIR}/parser
        ${CMAKE_CURRENT_SOURCE_DIR}/libs
)


add_executable(main
        neuro_model/main.cpp
        ${PROJECT_SOURCES}
)

target_include_directories(main PRIVATE
        ${PROJECT_INCLUDE_DIRS}
)

include(FetchContent)

set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.1
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cpp_neural_clustering
        bindings/pybind_module.cpp
        ${PROJECT_SOURCES}
)

target_include_directories(cpp_neural_clustering PRIVATE
        ${PROJECT_INCLUDE_DIRS}
)