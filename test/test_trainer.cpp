#include <gtest/gtest.h>
#include "../neuro_model/Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"

// ТЕСТ 1: Конструктор — проверка создания тренера
TEST(TrainerTest, ConstructorWorks) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 2, Activation::SIGMOID));
    net.addLayer(Layer(2, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    cfg.epochs = 1000;
    cfg.learning_rate = 0.1;
    
    Trainer trainer(net, cfg);
    
    SUCCEED();
}

// ТЕСТ 2: Обучение — не падает ли при запуске?
TEST(TrainerTest, TrainDoesNotCrash) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 3, Activation::RELU));
    net.addLayer(Layer(3, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    cfg.epochs = 5;
    cfg.learning_rate = 0.1;
    cfg.verbose = false;
    
    Trainer trainer(net, cfg);
    
    Matrix<double> inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;   
    inputs(1, 0) = 1.0; inputs(1, 1) = 1.0;   
    
    Matrix<double> targets(2, 1);
    targets(0, 0) = 0.0; 
    targets(1, 0) = 1.0; 
    
    EXPECT_NO_THROW(trainer.train(inputs, targets));
}

// ТЕСТ 3: Проверка возвращения числа
TEST(TrainerTest, PredictReturnsValue) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 2, Activation::SIGMOID));
    net.addLayer(Layer(2, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    Trainer trainer(net, cfg);
    
    std::vector<double> input = {0.5, -0.3};
    
    auto result = trainer.predict(input);
    
    ASSERT_FALSE(result.empty());
    EXPECT_GE(result[0], 0.0);
    EXPECT_LE(result[0], 1.0);
    EXPECT_TRUE(std::isfinite(result[0]));
}

// ТЕСТ 4: Ошибка уменьшается после обучения 
TEST(TrainerTest, LossDecreasesAfterTraining) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 4, Activation::RELU));
    net.addLayer(Layer(4, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    cfg.epochs = 100;    
    cfg.learning_rate = 0.5;
    cfg.verbose = false;
    
    Trainer trainer(net, cfg);
    
    // Подготавливаем данные для XOR
    Matrix<double> inputs(4, 2);
    Matrix<double> targets(4, 1);
    
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0; targets(0, 0) = 0.0;
    inputs(1, 0) = 0.0; inputs(1, 1) = 1.0; targets(1, 0) = 1.0;
    inputs(2, 0) = 1.0; inputs(2, 1) = 0.0; targets(2, 0) = 1.0;
    inputs(3, 0) = 1.0; inputs(3, 1) = 1.0; targets(3, 0) = 0.0;
    
    // Ошибка ДО обучения
    double error_before = trainer.compute_loss(inputs, targets);
    
    // Обучаем
    trainer.train(inputs, targets);
    
    // Ошибка ПОСЛЕ обучения
    double error_after = trainer.compute_loss(inputs, targets);
    
    // Ошибка должна уменьшиться
    EXPECT_LE(error_after, error_before);
}

// ТЕСТ 5: Оценка точности — возвращает процент [0; 100]
TEST(TrainerTest, EvaluateReturnsPercentage) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 3, Activation::RELU));
    net.addLayer(Layer(3, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    Trainer trainer(net, cfg);
    
    Matrix<double> inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;
    inputs(1, 0) = 1.0; inputs(1, 1) = 1.0;
    
    Matrix<double> targets(2, 1);
    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    
    double accuracy = trainer.evaluate(inputs, targets);
    
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 100.0);
    EXPECT_TRUE(std::isfinite(accuracy));
}

// ТЕСТ 6: Проверка обратного распространения с SIGMOID
TEST(TrainerTest, BackwardPassWithSigmoidWorks) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 3, Activation::SIGMOID));
    net.addLayer(Layer(3, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    Trainer trainer(net, cfg);
    
    Matrix<double> input(2, 1);
    input(0, 0) = 0.5;
    input(1, 0) = -0.3;
    
    Matrix<double> target(1, 1);
    target(0, 0) = 1.0;
    
    auto outputs = trainer.forward_pass(input);
    std::vector<Matrix<double>> deltas(net.getLayers().size());
    
    EXPECT_NO_THROW(trainer.backward_pass(outputs, target, deltas));
    
    // Проверяем, что градиенты не все нулевые
    bool has_nonzero = false;
    for (const auto& delta : deltas) {
        for (size_t i = 0; i < delta.rows(); ++i) {
            if (std::abs(delta(i, 0)) > 1e-10) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// ТЕСТ 7: Проверка predict_batch
TEST(TrainerTest, PredictBatchWorks) {
    NeuralNetwork net;
    net.addLayer(Layer(2, 2, Activation::SIGMOID));
    net.addLayer(Layer(2, 1, Activation::SIGMOID));
    
    TrainingConfig cfg;
    Trainer trainer(net, cfg);
    
    Matrix<double> inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;
    inputs(1, 0) = 1.0; inputs(1, 1) = 1.0;
    
    auto results = trainer.predict_batch(inputs);
    
    EXPECT_EQ(results.rows(), 2);
    EXPECT_EQ(results.cols(), 1);
    
    for (size_t i = 0; i < results.rows(); ++i) {
        EXPECT_GE(results(i, 0), 0.0);
        EXPECT_LE(results(i, 0), 1.0);
    }
}

// ТЕСТ 8: Проверка разных функций активации
TEST(TrainerTest, DifferentActivationsWork) {
    // Сеть с RELU и LINEAR
    NeuralNetwork net;
    net.addLayer(Layer(2, 3, Activation::RELU));
    net.addLayer(Layer(3, 1, Activation::LINEAR));
    
    TrainingConfig cfg;
    cfg.epochs = 10;
    cfg.learning_rate = 0.01;
    cfg.verbose = false;
    
    Trainer trainer(net, cfg);
    
    Matrix<double> inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;
    inputs(1, 0) = 1.0; inputs(1, 1) = 1.0;
    
    Matrix<double> targets(2, 1);
    targets(0, 0) = 0.0;
    targets(1, 0) = 2.0;
    
    EXPECT_NO_THROW(trainer.train(inputs, targets));
}
