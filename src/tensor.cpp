#include "tensor.hpp"
#include <numeric>
#include <algorithm>

Tensor::Tensor(std::vector<size_t> shape, double initial_value, bool requires_grad) : shape(shape) , requires_grad(requires_grad) {
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

void Tensor::zero_grad() {
    grad->fill(0.0);
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

std::shared_ptr<Tensor> Tensor::add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape != b->shape) throw std::invalid_argument("Shapes must match");

    bool req_grad = a->requires_grad || b->requires_grad;
    auto result = std::make_shared<Tensor>(a->shape, 0.0, req_grad);

    for (size_t i = 0; i < a->data.size(); i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    if (req_grad) {
        result->prev = {a, b};

        result->_backward = [a, b, result]() {
            if (a->requires_grad) {
                for (size_t i = 0; i < a->grad->data.size(); i++) {
                    a->grad->data[i] += result->grad->data[i];
                }
            }
            if (b->requires_grad) {
                for (size_t i = 0; i < b->grad->data.size(); i++) {
                    b->grad->data[i] += result->grad->data[i];
                }
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::matrixmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    std::shared_ptr<Tensor> result = a->matmul(b);

    bool req_grad = a->requires_grad || b->requires_grad;

    if (req_grad) {
        result->requires_grad = true;
        result->grad = std::make_shared<Tensor>(result->shape, 0.0, false);

        result->prev = {a, b};

        result->_backward = [a, b, result]() {
            if (a->requires_grad) {
                auto d_a = result->grad->matmul(b->transpose());
                *(a->grad) += *d_a;
            }
            if (b->requires_grad) {
                auto d_b = a->transpose()->matmul(result->grad);
                *(b->grad) += *d_b;
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::relu(std::shared_ptr<Tensor> a) {
    auto result = a->relu();

    if (a->requires_grad) {
        result->requires_grad = true;
        result->grad = std::make_shared<Tensor>(result->shape, 0.0, false);

        result->prev = {a};

        result->_backward = [a, result]() {
            for (size_t i = 0; i < a->data.size(); ++i) {
                double local_grad = (a->data[i] > 0.0) ? 1.0 : 0.0;
                a->grad->data[i] += result->grad->data[i] * local_grad;
            }
        };
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::transpose() const {
    if (this->shape.size() != 2) throw std::invalid_argument("Size must be 2");

    std::vector<size_t> new_shape = this->shape;
    std::reverse(new_shape.begin(), new_shape.end());

    auto result = std::make_shared<Tensor>(new_shape, 0.0,
                                             this->requires_grad);

    for (size_t i = 0; i < this->shape[0]; i++) {
        for (size_t j = 0; j < this->shape[1]; j++) {
            (*result)({j, i}) = (*this)({i, j});
        }
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) const {
    if (shape.size() != 2 || other->shape.size() != 2)
        throw std::invalid_argument("matmul is for 2D tensors only");
    if (shape[1] != other->shape[0])
        throw std::invalid_argument("Inner dimensions must match");

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other->shape[1];
    auto d = {M, N};
    auto result = std::make_shared<Tensor>(d, 0.0);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                sum += (*this)({i, k}) * (*other)({k, j});
            }
            (*result)({i, j}) = sum;
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

std::shared_ptr<Tensor> Tensor::apply(std::function<double(double)> func) const {
    auto result = std::make_shared<Tensor>(*this);
    for (double& val : result->data) {
        val = func(val);
    }
    return result;
}

std::shared_ptr<Tensor> Tensor::relu() const {
    auto result = std::make_shared<Tensor>(*this);
    result->apply_([](double val) {return val > 0.0 ? val : 0.0;});
    return result;
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

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(shared_from_this(), topo, visited);

    for (auto& t : topo) {
        if (t->grad) {
            t->zero_grad();
        }
    }

    if (this->grad) {
        this->grad->fill(1.0);
    }

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }
}
