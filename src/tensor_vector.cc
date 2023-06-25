#include "tensor_vector.hh"

void TensorVector::print() {
  // Tensor to std::vector
  // IMPORTANT: It must be on the CPU, not CUDA!
  // Stick with float for now
  auto a =
      torch::rand({3, 2}, torch::TensorOptions(torch::kCPU).dtype(torch::kF32));
  std::vector<float> a_vec(a.data_ptr<float>(),
                           a.data_ptr<float>() + a.numel());
  std::cout << a_vec << std::endl;

  // Add'l resources!!
  // https://www.simonwenkel.com/notes/software_libraries/pytorch/data_transfer_to_and_from_pytorch.html
}