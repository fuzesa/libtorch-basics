#include "cuda_basics.hh"

void CudaBasics::print() {
  // Check if a CUDA device is available
  const auto is_cuda_available = torch::cuda::is_available();
  const auto result = is_cuda_available ? "true" : "false";
  std::cout << "CUDA available? " << result << std::endl;

  // Get the number of CUDA devices
  const auto cuda_device_count = torch::cuda::device_count();
  std::cout << "CUDA device count: " << cuda_device_count << std::endl;

  // Get the current CUDA device index
  // const auto cuda_device_index = torch::device(torch::kCUDA).index();
  // std::cout << "CUDA device index: " << +cuda_device_index << std::endl;

  // Get the name of a CUDA device
  // const auto cuda_device_name =
  //     torch::cuda::getDeviceName(torch::cuda::current_device());
  // std::cout << "CUDA device name: " << cuda_device_name << std::endl;

  // Get the current CUDA device
  torch::Device current_cuda_device(torch::kCUDA);
  std::cout << "Current CUDA device: " << +current_cuda_device.index()
            << std::endl;
}