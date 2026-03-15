#pragma once

#include "tensor.hpp"
#include "optimizer.hpp"

class Linear {
private:
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> B;

public:
    Linear(size_t in_features, size_t out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);

    std::vector<std::shared_ptr<Tensor>> parameters()
};