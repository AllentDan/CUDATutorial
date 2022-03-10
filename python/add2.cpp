#include "add2.h"
#include <torch/extension.h>

void torch_launch_add2(torch::Tensor &c, const torch::Tensor &a,
                       const torch::Tensor &b, int64_t n) {
  launch_add2((float *)c.data_ptr(), (const float *)a.data_ptr(),
              (const float *)b.data_ptr(), n);
}

// a must for jit compile and a must for setup running
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_add2", &torch_launch_add2, "add2 kernel warpper");
}

// a must for cmake when running
TORCH_LIBRARY(add2, m) { m.def("torch_launch_add2", torch_launch_add2); }
