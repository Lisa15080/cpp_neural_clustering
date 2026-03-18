#include <gtest/gtest.h>
#include "../neuro_model/Trainer_class/trainer.h"
#include "../class/Matrix/matrix.h"

// ТЕСТ 1: Конструктор — проверка создания тренера
TEST(TrainerTest, ConstructorWorks) {
    NeuralNetwork net({2, 2, 1}, false);
    
    // Создаём тренера для этой сети
    Trainer trainer(net);
    
    // Проверяем, что тренер создался 
    auto config = trainer.getConfig();
    EXPECT_EQ(config.epochs, 1000);         
    EXPECT_DOUBLE_EQ(config.learning_rate, 0.1);
}

// ТЕСТ 2: Обучение — не падает ли при запуске?
TEST(TrainerTest, TrainDoesNotCrash) {
    NeuralNetwork net({2, 3, 1}, false);
    Trainer trainer(net);
    
    Matrix<double> inputs(2, 2);
    inputs(0, 0) = 0.0; inputs(0, 1) = 0.0;   
    inputs(1, 0) = 1.0; inputs(1, 1) = 1.0;   
    
    Matrix<double> targets(2, 1);
    targets(0, 0) = 0.0; 
    targets(1, 0) = 1.0; 
    
    TrainingConfig cfg;
    cfg.epochs = 5;
    cfg.verbose = false;
    trainer.setConfig(cfg);
    
    EXPECT_NO_THROW(trainer.train(inputs, targets));
}

// ТЕСТ 3: Проверка возвращения числа
TEST(TrainerTest, PredictReturnsValue) {
    NeuralNetwork net({2, 2, 1}, false);
    Trainer trainer(net);
    
    std::vector<double> input = {0.5, -0.3};
    
    auto result = trainer.predict(input);
    
    ASSERT_FALSE(result.empty());
    EXPECT_GE(result[0], 0.0);
    EXPECT_LE(result[0], 1.0);
    EXPECT_TRUE(std::isfinite(result[0]));
}

// ТЕСТ 4: Ошибка уменьшается после обучения 
TEST(TrainerTest, LossDecreasesAfterTraining) {
    NeuralNetwork net({2, 2, 1}, false);
    Trainer trainer(net);
    
    Matrix<double> input(2, 1);
    input(0, 0) = 0.5;
    input(1, 0) = 0.5;
    
    Matrix<double> target(1, 1);
    target(0, 0) = 1.0;
    
    // Ошибка ДО обучения
    auto out_before = trainer.predict(input);
    double error_before = (out_before(0, 0) - 1.0) * (out_before(0, 0) - 1.0);
    
    // Обучаем через публичный метод train()
    Matrix<double> inputs(1, 2);
    inputs(0, 0) = 0.5;
    inputs(0, 1) = 0.5;
    
    Matrix<double> targets(1, 1);
    targets(0, 0) = 1.0;
    
    TrainingConfig cfg;
    cfg.epochs = 50;    
    cfg.verbose = false;
    trainer.setConfig(cfg);
    
    trainer.train(inputs, targets);
    
    // Ошибка ПОСЛЕ обучения
    auto out_after = trainer.predict(input);
    double error_after = (out_after(0, 0) - 1.0) * (out_after(0, 0) - 1.0);
    
    // Ошибка должна уменьшиться (или остаться той же)
    // +1e-6 — допуск на численную погрешность
    EXPECT_LE(error_after, error_before + 1e-6);
}

// ТЕСТ 5: Оценка точности — возвращает процент [0; 100]
TEST(TrainerTest, EvaluateReturnsPercentage) {
    NeuralNetwork net({2, 2, 1}, false);
    Trainer trainer(net);
    
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
