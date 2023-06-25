#include "slicing_reshaping.hh"

void SlicingReshaping::print() {
  auto tensor1 = torch::arange(8);
  std::cout << tensor1 << std::endl;

  // Print shape of tensor
  std::cout << tensor1.sizes() << std::endl;  // Would be shape in python

  // Reshape tensor using view
  // view works ONLY on contigous tensors and doesn't copy the data
  // A contiguous tensor is a tensor whose elements are stored in a contiguous
  // order without leaving any empty space between them
  auto tensor2 = tensor1.view({4, 2});
  std::cout << tensor2 << std::endl;

  // Reshape tensor using reshape
  // reshape works on any tensor and copies the data if it's necessary
  auto tensor3 = tensor1.reshape({2, 4});
  std::cout << tensor3 << std::endl;

  // Slicing
  auto tensor4 = torch::arange(8).reshape({4, 2});
  auto val = tensor4[2][1].item();
  std::cout << val << std::endl;
}