#include "tensor.hpp"

int main() {
    try {
        Tensor a({1}, 10.0, true);
        Tensor b({1}, 5.0, true);
        Tensor c = a + b;

        c.get_grad()->get_data()[0] = 1.0;
        c._backward();

        std::cout << a.get_grad()->get_data()[0] << std::endl;
        std::cout << b.get_grad()->get_data()[0] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
