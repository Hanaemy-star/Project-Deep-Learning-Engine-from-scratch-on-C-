#include "tensor.hpp"
#include <numeric>

Tensor::Tensor(std::vector<size_t> shape, double initial_value) : shape(shape) {
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

Tensor& Tensor::operator+=(const Tensor& other) {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for in-place addition");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("Shapes must match for in-place addition");
    }
    Tensor result = *this;
    result += other;
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
            std::cout << this({i, j}) << "\t";
        }
        std::cout << std::endl;
    }
}