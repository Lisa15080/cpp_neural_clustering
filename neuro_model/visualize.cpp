// visualize.cpp
#include "visualize.h"
#include <fstream>
#include <vector>

void save_true_clusters_json(const std::string& filename,
                             const Matrix<double>& inputs,
                             const Matrix<double>& targets) {
    std::ofstream f(filename);
    if (!f.is_open()) return;

    f << "{\"x\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        f << inputs(i, 0);
    }
    f << "], \"y\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        f << inputs(i, 1);
    }
    f << "], \"labels\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        f << static_cast<int>(targets(i, 0) > 0.5 ? 1 : 0);
    }
    f << "]}";
}

void save_predictions_json(const std::string& filename,
                           const Matrix<double>& inputs,
                           Trainer& trainer) {
    std::ofstream f(filename);
    if (!f.is_open()) return;

    f << "{\"x\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        f << inputs(i, 0);
    }
    f << "], \"y\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        f << inputs(i, 1);
    }
    f << "], \"labels\": [";
    for(size_t i = 0; i < inputs.rows(); ++i) {
        if(i) f << ", ";
        std::vector<double> pt = {inputs(i, 0), inputs(i, 1)};
        double prob = trainer.predict(pt)[0];
        f << static_cast<int>(prob > 0.5 ? 1 : 0);
    }
    f << "]}";
}