#pragma once

#include <iostream>
#include <vector>
#include <functional>

class Tensor {
private:
    std::vector<double> data;
    std::vector<size_t> shape;
    size_t calculate_size(const std::vector<size_t>& s);

public:
    Tensor(std::vector<size_t> shape, double initial_value = 0.0);

    double& operator()(const std::vector<size_t>& indices);

    double operator()(const std::vector<size_t>& indices) const;

    void reshape(std::vector<size_t> nshape);

    Tensor& operator+=(const Tensor& other);

    Tensor operator+(const Tensor& other) const;

    Tensor matmul(const Tensor& other) const;

    void print() const;

    Tensor& apply_(std::function<double(double)> func);

    Tensor apply(std::function<double(double)> func) const;
};