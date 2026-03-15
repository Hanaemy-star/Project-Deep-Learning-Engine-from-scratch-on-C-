#include "linear.hpp"
#include <random>

Linear::Linear(size_t in_features, size_t out_features) {
    W = std::make_shared<Tensor>(std::vector<size_t>{in_features, out_features}, 0.0, true);
    B = std::make_shared<Tensor>(std::vector<size_t>{1, out_features}, 0.0, true);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, 0.1);

    for (auto& v : W->get_data()) {
        double val = dis(gen);
        v = val;
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto Y = Tensor::add(Tensor::matrixmul(input, this->W), this->B);
    return Y;
}

std::vector<std::shared_ptr<Tensor> > Linear::parameters() {
    return {W, B};
}
