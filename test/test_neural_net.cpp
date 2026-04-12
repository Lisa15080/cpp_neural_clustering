#include <gtest/gtest.h>
#include "../neuro_model/Neural_Net/neural_net.h"
#include <chrono>
#include <fstream>
#include <cmath>
#include <cstdio>

// Тест 1: Конструктор создаёт слои правильных размеров
TEST(NeuralNetworkTest, ConstructorCreatesCorrectLayers) {
    std::vector<int> sizes = {2, 3, 1};
    NeuralNetwork net(sizes, Activation::SIGMOID, false);

    auto& layers = net.getLayers();
    ASSERT_EQ(layers.size(), 2u);

    // Первый слой: 2 входа -> 3 нейрона
    EXPECT_EQ(layers[0].weights.rows(), 3u);
    EXPECT_EQ(layers[0].weights.cols(), 2u);
    EXPECT_EQ(layers[0].biases.rows(), 3u);
    EXPECT_EQ(layers[0].biases.cols(), 1u);
    EXPECT_EQ(layers[0].activation, Activation::SIGMOID);

    // Второй слой: 3 входа -> 1 нейрон (последний всегда SIGMOID)
    EXPECT_EQ(layers[1].weights.rows(), 1u);
    EXPECT_EQ(layers[1].weights.cols(), 3u);
    EXPECT_EQ(layers[1].biases.rows(), 1u);
    EXPECT_EQ(layers[1].biases.cols(), 1u);
    EXPECT_EQ(layers[1].activation, Activation::SIGMOID);
}

// Тест 2: Проверка функций активации и их производных
TEST(NeuralNetworkTest, ActivationFunctions) {
    NeuralNetwork net({1, 1}, Activation::SIGMOID, false);

    // Сигмоида
    EXPECT_NEAR(net.sigmoid(0.0), 0.5, 1e-9);
    EXPECT_NEAR(net.sigmoid(1.0), 0.7310585786300049, 1e-9);
    EXPECT_NEAR(net.sigmoid(-1.0), 0.2689414213699951, 1e-9);

    // Производная сигмоиды: s*(1-s)
    double x = 0.7;
    double s = net.sigmoid(x);
    EXPECT_NEAR(net.sigmoidDerivative(x), s * (1.0 - s), 1e-9);

    // ReLU
    EXPECT_DOUBLE_EQ(net.relu(5.0), 5.0);
    EXPECT_DOUBLE_EQ(net.relu(-2.5), 0.0);
    EXPECT_DOUBLE_EQ(net.relu(0.0), 0.0);

    // Производная ReLU
    EXPECT_DOUBLE_EQ(net.reluDerivative(5.0), 1.0);
    EXPECT_DOUBLE_EQ(net.reluDerivative(-2.5), 0.0);
    EXPECT_DOUBLE_EQ(net.reluDerivative(0.0), 0.0);

    // Linear
    EXPECT_DOUBLE_EQ(net.linear(42.0), 42.0);
    EXPECT_DOUBLE_EQ(net.linearDerivative(0.0), 1.0);
}

// Тест 3: Прямой проход с фиксированными весами (сравнение с ручным расчётом)
TEST(NeuralNetworkTest, ForwardWithFixedWeights) {
    NeuralNetwork net({2, 2, 1}, Activation::SIGMOID, false);
    auto& layers = net.getLayers();

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
    std::vector<double> output = net.forward(input);

    // Ручной расчёт:
    // Скрытый слой: z0 = 0.5*1 + 0.5*0.5 + 0.1 = 0.85
    //               z1 = 0.5*1 + 0.5*0.5 + 0.1 = 0.85
    // Активации (сигмоида): a0 = a1 = sigmoid(0.85) ≈ 0.700567
    // Выходной слой: z_out = 0.7*a0 + 0.3*a1 + 0.2 = 1.0*a + 0.2 ≈ 0.900567
    // Выход: sigmoid(0.900567) ≈ 0.71095
    double a = 1.0 / (1.0 + std::exp(-0.85));
    double z_out = a + 0.2;
    double expected = 1.0 / (1.0 + std::exp(-z_out));

    EXPECT_NEAR(output[0], expected, 1e-5);
}

// Тест 4: Сохранение и загрузка модели (веса должны совпадать)
TEST(NeuralNetworkTest, SaveAndLoadModel) {
    const std::string filename = "test_model_tmp.txt";

    // Создаём сеть со случайными весами и сохраняем
    NeuralNetwork net1({2, 3, 1}, Activation::RELU, false);
    bool saveOk = net1.saveModel(filename);
    ASSERT_TRUE(saveOk);

    // Загружаем в другую сеть
    NeuralNetwork net2({2, 3, 1}, Activation::RELU, false);
    bool loadOk = net2.loadModel(filename);
    ASSERT_TRUE(loadOk);

    auto& layers1 = net1.getLayers();
    auto& layers2 = net2.getLayers();
    ASSERT_EQ(layers1.size(), layers2.size());

    // Послойное сравнение весов и смещений
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
        // Тип активации тоже должен сохраниться (RELU на скрытом, SIGMOID на выходе)
        EXPECT_EQ(layers1[l].activation, layers2[l].activation);
    }

    std::remove(filename.c_str());
}

// Тест 5: Загрузка несуществующего файла возвращает false
TEST(NeuralNetworkTest, LoadNonExistentFileFails) {
    NeuralNetwork net({2, 2}, Activation::SIGMOID, false);
    bool result = net.loadModel("file_that_does_not_exist_12345.txt");
    EXPECT_FALSE(result);
}

// Тест 6: Сигмоида при больших по модулю аргументах насыщается
TEST(NeuralNetworkTest, SigmoidSaturation) {
    NeuralNetwork net({1, 1}, Activation::SIGMOID, false);
    EXPECT_NEAR(net.sigmoid(1e6), 1.0, 1e-6);
    EXPECT_NEAR(net.sigmoid(-1e6), 0.0, 1e-6);
}

// Тест 7: Случайная инициализация весов лежит в диапазоне [-1, 1]
TEST(NeuralNetworkTest, RandomInitializationRange) {
    NeuralNetwork net({5, 10, 5}, Activation::SIGMOID, false);
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

// Тест 8: Добавление слоя динамически
TEST(NeuralNetworkTest, AddLayer) {
    NeuralNetwork net({2, 3}, Activation::SIGMOID, false); // только входной и выходной слои (2 -> 3)
    EXPECT_EQ(net.numLayers(), 1u); // один слой весов

    net.addLayer(4, Activation::RELU);
    EXPECT_EQ(net.numLayers(), 2u);

    auto& layers = net.getLayers();
    // Первый слой остался 2->3, сигмоида
    EXPECT_EQ(layers[0].weights.rows(), 3u);
    EXPECT_EQ(layers[0].weights.cols(), 2u);
    EXPECT_EQ(layers[0].activation, Activation::SIGMOID);
    // Новый слой: 3->4, ReLU
    EXPECT_EQ(layers[1].weights.rows(), 4u);
    EXPECT_EQ(layers[1].weights.cols(), 3u);
    EXPECT_EQ(layers[1].activation, Activation::RELU);

    // Проверяем, что прямой проход работает
    std::vector<double> input = {0.5, -0.3};
    std::vector<double> output = net.forward(input);
    EXPECT_EQ(output.size(), 4u);
}

// Тест 9: Предсказание вероятностей для бинарного случая

TEST(NeuralNetworkTest, PredictProbabilities) {
    NeuralNetwork net({2, 1}, Activation::SIGMOID, false);
    auto& layers = net.getLayers();
    // Установим веса так, чтобы для входа {1,1} выход был 0.8
    layers[0].weights(0, 0) = 1.0;
    layers[0].weights(0, 1) = 1.0;
    layers[0].biases(0, 0) = 0.0; // z = 2, sigmoid ≈ 0.8808, но давайте подберём для 0.8

    // Для простоты используем predictProbabilities
    std::vector<double> probs = net.predictProbabilities({0.0, 0.0});
    EXPECT_EQ(probs.size(), 2u);
    EXPECT_NEAR(probs[0] + probs[1], 1.0, 1e-9);

    // predictProba должен вернуть вероятность класса 1
    double p1 = net.predictProba({0.0, 0.0});
    EXPECT_NEAR(p1, probs[1], 1e-9);
}

// Тест 10: Производительность прямого прохода
TEST(NeuralNetworkTest, ForwardPerformance) {
    NeuralNetwork net({100, 200, 50, 10}, Activation::RELU, false);
    std::vector<double> input(100, 0.5);

    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        net.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[INFO] " << iterations << " forward passes took " << duration.count() << " ms" << std::endl;
    // Тест всегда проходит, просто информационный вывод
    SUCCEED();
}

// main для запуска тестов
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}