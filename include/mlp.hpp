#pragma once

#include "linear.hpp"

class MLP {
private:
    Linear layer1;
    Linear layer2;
public:
    MLP(size_t in_features, size_t hidden_features, size_t out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    std::vector<std::shared_ptr<Tensor>> parameters();
};