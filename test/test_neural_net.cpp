#include <gtest/gtest.h>
#include "../neuro_model/Neural_Net/neural_net.h"
#include <chrono>
#include <fstream>

// Тест 1: проверяем, что конструктор создаёт слои правильных размеров
TEST(NeuralNetworkTest, ConstructorCreatesCorrectLayers) {
    std::vector<int> sizes = {2, 3, 1};
    NeuralNetwork net(sizes, false);               // логирование выключено

    auto& layers = net.getLayers();
    ASSERT_EQ(layers.size(), 2);                   // должно быть 2 слоя

    // Первый слой: 2 входа -> 3 нейрона
    EXPECT_EQ(layers[0].weights.rows(), 3);
    EXPECT_EQ(layers[0].weights.cols(), 2);
    EXPECT_EQ(layers[0].biases.rows(), 3);
    EXPECT_EQ(layers[0].biases.cols(), 1);

    // Второй слой: 3 входа -> 1 нейрон
    EXPECT_EQ(layers[1].weights.rows(), 1);
    EXPECT_EQ(layers[1].weights.cols(), 3);
    EXPECT_EQ(layers[1].biases.rows(), 1);
    EXPECT_EQ(layers[1].biases.cols(), 1);
}

// Тест 2: проверяем функции активации и их производные
TEST(NeuralNetworkTest, ActivationFunctions) {
    NeuralNetwork net({1,1}, false);   // сеть с одним слоем, просто чтобы вызвать методы

    // Сигмоида
    EXPECT_NEAR(net.sigmoid(0.0), 0.5, 1e-9);
    EXPECT_NEAR(net.sigmoid(1.0), 0.7310585786300049, 1e-9);
    EXPECT_NEAR(net.sigmoid(-1.0), 0.2689414213699951, 1e-9);

    // Производная сигмоиды должна равняться s*(1-s)
    double x = 0.7;
    double s = net.sigmoid(x);
    EXPECT_NEAR(net.sigmoidDerivative(x), s * (1 - s), 1e-9);

    // ReLU
    EXPECT_EQ(net.relu(5.0), 5.0);
    EXPECT_EQ(net.relu(-2.5), 0.0);
    EXPECT_EQ(net.relu(0.0), 0.0);

    // Производная ReLU
    EXPECT_EQ(net.reluDerivative(5.0), 1.0);
    EXPECT_EQ(net.reluDerivative(-2.5), 0.0);
    EXPECT_EQ(net.reluDerivative(0.0), 0.0);
}

// Тест 3: прямой проход с фиксированными весами (считаем вручную)
TEST(NeuralNetworkTest, ForwardWithFixedWeights) {
    // Сеть 2 -> 2 -> 1
    NeuralNetwork net({2, 2, 1}, false);
    auto& layers = net.getLayers();

    // Задаём веса и смещения вручную
    // Слой 0: все веса = 0.5, смещения = 0.1
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            layers[0].weights(i, j) = 0.5;
        }
        layers[0].biases(i, 0) = 0.1;
    }
    // Слой 1: веса [0.7, 0.3], смещение 0.2
    layers[1].weights(0, 0) = 0.7;
    layers[1].weights(0, 1) = 0.3;
    layers[1].biases(0, 0) = 0.2;

    std::vector<double> input = {1.0, 0.5};
    auto output = net.forward(input);

    // Ручной расчёт
    double hidden = 1.0 / (1.0 + exp(-0.85));   // сигмоида от 0.85
    double outPre = 0.7 * hidden + 0.3 * hidden + 0.2;   // = hidden + 0.2
    double expected = 1.0 / (1.0 + exp(-outPre));

    EXPECT_NEAR(output[0], expected, 1e-5);
}

// Тест 4: сохранение и загрузка модели (должны совпадать веса)
TEST(NeuralNetworkTest, SaveAndLoadModel) {
    const std::string filename = "test_model.txt";

    // Сохраняем первую сеть
    NeuralNetwork net1({2, 3, 1}, false);
    net1.saveModel(filename);

    // Загружаем во вторую сеть
    NeuralNetwork net2({2, 3, 1}, false);
    bool loadOk = net2.loadModel(filename);
    ASSERT_TRUE(loadOk);

    auto& layers1 = net1.getLayers();
    auto& layers2 = net2.getLayers();

    ASSERT_EQ(layers1.size(), layers2.size());

    // Сравниваем все веса и смещения с точностью 1e-6
    for (size_t l = 0; l < layers1.size(); ++l) {
        ASSERT_EQ(layers1[l].weights.rows(), layers2[l].weights.rows());
        ASSERT_EQ(layers1[l].weights.cols(), layers2[l].weights.cols());
        for (size_t i = 0; i < layers1[l].weights.rows(); ++i) {
            for (size_t j = 0; j < layers1[l].weights.cols(); ++j) {
                EXPECT_NEAR(layers1[l].weights(i, j), layers2[l].weights(i, j), 1e-6);
            }
        }
        ASSERT_EQ(layers1[l].biases.rows(), layers2[l].biases.rows());
        for (size_t i = 0; i < layers1[l].biases.rows(); ++i) {
            EXPECT_NEAR(layers1[l].biases(i, 0), layers2[l].biases(i, 0), 1e-6);
        }
    }

    std::remove(filename.c_str());   // удаляем временный файл
}

// Тест 5: попытка загрузить несуществующий файл должна вернуть false
TEST(NeuralNetworkTest, LoadNonExistentFileFails) {
    NeuralNetwork net({2,2}, false);
    bool result = net.loadModel("file_that_does_not_exist.txt");
    EXPECT_FALSE(result);
}

// Тест 6: сигмоида от очень больших чисел должна быть близка к 0 или 1
TEST(NeuralNetworkTest, SigmoidSaturation) {
    NeuralNetwork net({1,1}, false);
    EXPECT_NEAR(net.sigmoid(1e6), 1.0, 1e-6);
    EXPECT_NEAR(net.sigmoid(-1e6), 0.0, 1e-6);
}

// Тест 7: случайные веса должны лежать в диапазоне [-1, 1]
TEST(NeuralNetworkTest, RandomInitializationRange) {
    NeuralNetwork net({5, 10, 5}, false);
    auto& layers = net.getLayers();

    for (const auto& layer : layers) {
        for (size_t i = 0; i < layer.weights.rows(); ++i) {
            for (size_t j = 0; j < layer.weights.cols(); ++j) {
                double val = layer.weights(i, j);
                EXPECT_GE(val, -1.0);
                EXPECT_LE(val, 1.0);
            }
        }
        for (size_t i = 0; i < layer.biases.rows(); ++i) {
            double val = layer.biases(i, 0);
            EXPECT_GE(val, -1.0);
            EXPECT_LE(val, 1.0);
        }
    }
}

// Тест 8: замеряем время выполнения 100 прямых проходов (просто информация)
TEST(NeuralNetworkTest, ForwardPerformance) {
    NeuralNetwork net({100, 200, 50, 10}, false);
    std::vector<double> input(100, 0.5);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        net.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "100 forward passes took " << duration.count() << " ms" << std::endl;
}