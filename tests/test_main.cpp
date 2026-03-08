#include "tensor.hpp"

void run_test() {
    auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 2}, 0.0, false);
    X->get_data() = {0.5, -0.5};

    auto W = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, 0.0, true);
    W->get_data() = {2.0, 0.0, 0.0, 2.0};

    auto B = std::make_shared<Tensor>(std::vector<size_t>{1, 2}, 0.1, true);

    auto Z_mul = Tensor::matrixmul(X, W);
    auto Z = Tensor::add(Z_mul, B);
    auto A = Tensor::relu(Z);

    std::cout << "--- Forward Pass ---" << std::endl;
    std::cout << "Output A (Expected: 1.1, 0):" << std::endl;
    A->print();

    A->backward();

    std::cout << "\n--- Backward Pass ---" << std::endl;

    std::cout << "Gradient of W (Expected: [[0.5, 0], [-0.5, 0]]):" << std::endl;
    W->get_grad()->print();

    std::cout << "Gradient of B (Expected: [1, 0]):" << std::endl;
    B->get_grad()->print();
}

int main() {
    try {
        run_test();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
    }
    return 0;
}
