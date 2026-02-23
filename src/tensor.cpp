#include "tensor.hpp"
#include <numeric>


Tensor::Tensor(std::vector<size_t> shape, double initial_value, bool requires_grad) : shape(shape) {
    if (requires_grad) {
        grad = std::make_shared<Tensor>(shape, 0.0, false);
    }
    size_t total_size = calculate_size(shape);
    data = std::vector<double>(total_size, initial_value);
}

size_t Tensor::calculate_size(const std::vector<size_t>& s) {
    if (s.empty()) return 0;
    size_t res = 1;
    for (auto dim : s) res *= dim;
    return res;
}

double& Tensor::operator()(const std::vector<size_t>& indices) {
    size_t flat_index = 0;
    size_t strides = 1;
    for (int i = indices.size() - 1; i >= 0; i--) {
        flat_index += indices[i] * strides;
        strides *= shape[i];
    }
    return data[flat_index];
}

double Tensor::operator()(const std::vector<size_t>& indices) const {
    size_t flat_index = 0;
    size_t strides = 1;
    for (int i = indices.size() - 1; i >= 0; i--) {
        flat_index += indices[i] * strides;
        strides *= shape[i];
    }
    return data[flat_index];
}

void Tensor::reshape(std::vector<size_t> nshape) {
    if (calculate_size(nshape) != data.size()) {
        throw std::invalid_argument("New shape must have the same total number of elements");
    }
    shape = nshape;
}

void Tensor::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for in-place addition");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

std::vector<double>& Tensor::get_data() {
    return this->data;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for in-place addition");
    }
    Tensor result(shape, 0.0, (this->requires_grad || other.requires_grad));
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = this->data[i] + other.data[i];
    }
    if (result.requires_grad) {
        auto left_grad = this->get_grad();
        auto right_grad = other.get_grad();
        auto res_grad = result.get_grad();

        result._backward = [left_grad, right_grad, res_grad]() {
            if (left_grad) {
                for (size_t i = 0; i < left_grad->data.size(); i++) {
                    left_grad->data[i] += res_grad->data[i];
                }
            }
            if (right_grad) {
                for (size_t i = 0; i < right_grad->data.size(); i++) {
                    right_grad->data[i] += res_grad->data[i];
                }
            }
        };
        result.prev = {left_grad, right_grad};
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2)
        throw std::invalid_argument("matmul is for 2D tensors only");
    if (shape[1] != other.shape[0])
        throw std::invalid_argument("Inner dimensions must match");

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other.shape[1];

    Tensor result({M, N}, 0.0);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += (*this)({i, k}) * other({k, j});
            }
            result({i, j}) = sum;
        }
    }
    return result;
}

void Tensor::print() const {
    if (shape.size() != 2) {
        std::cout << "Printing for N-dim not implemented yet\n";
        return;
    }
    for (size_t i = 0; i < shape[0]; i++) {
        for (size_t j = 0; j < shape[1]; j++) {
            std::cout << (*this)({i, j}) << "\t";
        }
        std::cout << std::endl;
    }
}

Tensor& Tensor::apply_(std::function<double(double)> func) {
    for (double& val : data) {
        val = func(val);
    }
    return *this;
}

Tensor Tensor::apply(std::function<double(double)> func) const {
    Tensor result = *this;
    for (double& val : result.data) {
        val = func(val);
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result = *this;
    return result.apply_([](double val) {return val > 0.0 ? val : 0.0;});
}

Tensor Tensor::operator*(const double& scalar) const {
    Tensor result = *this;
    return result.apply_([scalar](double val) {return val * scalar;});
}

std::shared_ptr<Tensor> Tensor::get_grad() const {
    return this->grad;
}

void Tensor::build_topo(std::shared_ptr<Tensor> curr,
                    std::vector<std::shared_ptr<Tensor>>& topo,
                    std::unordered_set<Tensor*>& visited) {

    if (visited.find(curr.get()) != visited.end()) {
        return;
    }

    visited.insert(curr.get());

    for (auto& parent : curr->prev) {
        build_topo(parent, topo, visited);
    }
    topo.push_back(curr);
}