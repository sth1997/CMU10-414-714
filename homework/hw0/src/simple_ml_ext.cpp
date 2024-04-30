#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *X, const float *Y, float *Z, size_t m, size_t dim, size_t n)
{
    size_t idx = 0;
    for (size_t i = 0; i < m; ++i)
        for (size_t j =0; j < n; ++j)
        {
            Z[idx] = 0.0f;
            for (size_t k = 0; k < dim; ++k)
                Z[idx] += X[i * dim + k] * Y[k * n + j];
            ++idx;
        }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t total_batch_num = (m + batch - 1) / batch;
    float* Z = new float[batch * k];
    float* X_T = new float[n * batch];
    float* tmp_mat = new float[n * k];
    for (size_t batch_num = 0; batch_num < total_batch_num; ++batch_num)
    {
        auto X_batch = X + batch_num * batch * n;
        auto y_batch = y + batch_num * batch;
        matmul(X_batch, theta, Z, batch, n, k);
        // exp
        for (size_t i = 0; i < batch * k; ++i)
            Z[i] = exp(Z[i]);
        //normalize
        for (size_t i = 0; i < batch; ++i)
        {
            float *tmp_Z = Z + i * k;
            float sum = 0.0f;
            for (size_t j = 0; j < k; ++j)
                sum += tmp_Z[j];
            for (size_t j = 0; j < k; ++j)
                tmp_Z[j] /= sum;
        }
        // Z - I_y
        for (size_t i = 0; i < batch; ++i)
            Z[i * k + y_batch[i]] -= 1.0;
        //X_batch.T @ (Z-I_y)
        size_t idx = 0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < batch; ++j)
            {
                X_T[idx] =  X_batch[j * n + i];
                ++idx;
            }
        matmul(X_T, Z, tmp_mat, n, batch, k);
        for (size_t i = 0; i < n * k; ++i)
            theta[i] -= lr / batch * tmp_mat[i];

    }
    delete[] Z;
    delete[] X_T;
    delete[] tmp_mat;

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
