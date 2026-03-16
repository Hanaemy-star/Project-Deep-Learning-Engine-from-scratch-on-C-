#include "mlp.hpp"

MLP::MLP(size_t in_features, size_t hidden_features, size_t out_features) : layer1(in_features, hidden_features),
      layer2(hidden_features, out_features) {}

std::shared_ptr<Tensor> MLP::forward(std::shared_ptr<Tensor> input) {
    auto hidden = layer1.forward(input);
    auto activated = hidden->leaky_relu();
    auto output = layer2.forward(activated);
    return output;
}

std::vector<std::shared_ptr<Tensor>> MLP::parameters() {
    std::vector<std::shared_ptr<Tensor>> all_params;

    auto p1 = layer1.parameters();
    auto p2 = layer2.parameters();

    all_params.insert(all_params.end(), p1.begin(), p1.end());
    all_params.insert(all_params.end(), p2.begin(), p2.end());

    return all_params;
}