#include <ATen/cuda/CUDAContext.h>

#include <chrono>

#include "base_class.hh"

class CudaBasics : public BaseClass {
 public:
  void print() override;
};