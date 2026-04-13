// visualize.h
#pragma once

#include "../class/Matrix/matrix.h"
#include "Trainer_class/trainer.h"
#include <string>

void save_true_clusters_json(const std::string& filename,
                             const Matrix<double>& inputs,
                             const Matrix<double>& targets);

void save_predictions_json(const std::string& filename,
                           const Matrix<double>& inputs,
                           Trainer& trainer);