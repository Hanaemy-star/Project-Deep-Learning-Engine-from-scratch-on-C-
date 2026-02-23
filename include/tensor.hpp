#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_set>

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::vector<double> data;
    std::vector<size_t> shape;
    std::shared_ptr<Tensor> grad;
    std::vector<std::shared_ptr<Tensor>> prev;
    bool requires_grad;

    size_t calculate_size(const std::vector<size_t>& s);
    void zero_grad();

public:

    std::function<void()> _backward;

    Tensor(std::vector<size_t> shape, double initial_value = 0.0, bool requires_grad = false);

    double& operator()(const std::vector<size_t>& indices);

    double operator()(const std::vector<size_t>& indices) const;

    void reshape(std::vector<size_t> nshape);

    void fill(double value);

    std::shared_ptr<Tensor> get_grad() const;

    std::vector<double>& get_data();

    Tensor& operator+=(const Tensor& other);

    Tensor operator+(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;

    void print() const;

    Tensor& apply_(std::function<double(double)> func);

    Tensor apply(std::function<double(double)> func) const;

    Tensor relu() const;

    Tensor operator*(const double& scalar) const;

    void build_topo(std::shared_ptr<Tensor> curr,
                    std::vector<std::shared_ptr<Tensor>>& topo,
                    std::unordered_set<Tensor*>& visited);

    void backward();
};