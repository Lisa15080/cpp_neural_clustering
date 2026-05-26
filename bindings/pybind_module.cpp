#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../neuro_model/Neural_Net/neural_net.h"
#include "../neuro_model/Trainer_class/trainer.h"
#include "../neuro_model/DataSet/dataset.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_neural_clustering, m) {
    m.doc() = "Python wrapper for cpp_neural_clustering";

    py::enum_<Activation>(m, "Activation")
        .value("SIGMOID", Activation::SIGMOID)
        .value("RELU", Activation::RELU)
        .value("LINEAR", Activation::LINEAR);

    py::class_<TrainingConfig>(m, "TrainingConfig")
        .def(py::init<>())
        .def_readwrite("epochs", &TrainingConfig::epochs)
        .def_readwrite("learning_rate", &TrainingConfig::learning_rate)
        .def_readwrite("verbose", &TrainingConfig::verbose);

    py::class_<Dataset>(m, "Dataset")
        .def(py::init<>())
        .def_readwrite("inputs", &Dataset::inputs)
        .def_readwrite("targets", &Dataset::targets);

    py::class_<DatasetGenerator>(m, "DatasetGenerator")
        .def(py::init<>())
        .def("generate_gaussian", &DatasetGenerator::generate_gaussian,
             py::arg("n_samples") = 100,
             py::arg("cluster_std") = 0.5,
             py::arg("separation") = 2.0);

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int>&, Activation, bool, const std::string&>(),
             py::arg("sizes"),
             py::arg("hidden_activation") = Activation::SIGMOID,
             py::arg("enable_logging") = false,
             py::arg("log_filename") = "")
        .def("forward", &NeuralNetwork::forward)
        .def("predict", &NeuralNetwork::predict)
        .def("predict_proba", &NeuralNetwork::predictProba)
        .def("predict_probabilities", &NeuralNetwork::predictProbabilities)
        .def("save_model", &NeuralNetwork::saveModel)
        .def("load_model", &NeuralNetwork::loadModel)
        .def("num_layers", &NeuralNetwork::numLayers)
        .def("input_size", &NeuralNetwork::inputSize)
        .def("output_size", &NeuralNetwork::outputSize);

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<NeuralNetwork&, const TrainingConfig&>(),
             py::arg("network"),
             py::arg("config") = TrainingConfig(),
             py::keep_alive<1, 2>())
        .def("train",
             py::overload_cast<
                 const std::vector<std::vector<double>>&,
                 const std::vector<std::vector<double>>&
             >(&Trainer::train))
        .def("evaluate",
             py::overload_cast<
                 const std::vector<std::vector<double>>&,
                 const std::vector<std::vector<double>>&
             >(&Trainer::evaluate, py::const_))
        .def("predict",
             py::overload_cast<const std::vector<double>&>(&Trainer::predict, py::const_))
        .def("predict_class", &Trainer::predict_class);
}