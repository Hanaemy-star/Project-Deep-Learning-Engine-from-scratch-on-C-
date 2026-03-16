#include <iostream>
#include <vector>
#include <memory>
#include <iomanip>
#include "tensor.hpp"
#include "mlp.hpp"
#include "optimizer.hpp"

int main() {
    std::cout << "=== Starting XOR Training Test ===" << std::endl;

    // 1. Prepare XOR Data
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    // Targets: 0, 1, 1, 0
    std::vector<double> targets = {0.0, 1.0, 1.0, 0.0};

    // 2. Initialize Model: 2 inputs -> 4 hidden neurons -> 1 output
    MLP model(2, 16, 1);

    // 3. Setup Optimizer (SGD)
    double lr = 0.1; // Slightly higher learning rate for XOR
    auto optimizer = std::make_shared<Optimizer>(model.parameters(), lr);

    // 4. Training Loop
    const int epochs = 5000;
    std::cout << "Training for " << epochs << " epochs..." << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        optimizer->zero_grad();

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Create tensors for the current sample
            auto x_tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 2}, 0.0, false);
            x_tensor->get_data() = inputs[i];

            auto y_target = std::make_shared<Tensor>(std::vector<size_t>{1, 1}, targets[i], false);

            // Forward Pass
            auto y_pred = model.forward(x_tensor);

            // Loss Calculation
            auto loss = Tensor::mse_loss(y_pred, y_target);
            total_loss += loss->get_data()[0];

            // Backward Pass
            loss->backward();
        }

        // Update weights
        optimizer->step();

        // Print progress
        if (epoch % 500 == 0) {
            std::cout << "Epoch: " << std::setw(5) << epoch
                      << " | Avg Loss: " << std::fixed << std::setprecision(6) << total_loss / 4.0 << std::endl;
        }
    }

    // 5. Final Evaluation
    std::cout << "\n=== Final Predictions ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto x_tensor = std::make_shared<Tensor>(std::vector<size_t>{1, 2}, 0.0, false);
        x_tensor->get_data() = inputs[i];

        auto result = model.forward(x_tensor);

        std::cout << "In: (" << inputs[i][0] << ", " << inputs[i][1] << ") "
                  << "Target: " << targets[i]
                  << " | Pred: " << std::fixed << std::setprecision(4) << result->get_data()[0] << std::endl;
    }

    return 0;
}