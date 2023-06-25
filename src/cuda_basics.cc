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
  torch::Device current_cuda_device_one(torch::kCUDA, 0);

  /* std::cout << "Current CUDA device: " << +current_cuda_device_one.index()
            << std::endl; */

  const auto props = at::cuda::getCurrentDeviceProperties();
  std::cout << "Current CUDA device: " << props->name << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  std::ios_base::sync_with_stdio(false);

  for (int i = 0; i < 1000; i++) {
    auto tensor_one = torch::rand({1000, 1000}, current_cuda_device_one);
    auto tensor_two = torch::rand({1000, 1000}, current_cuda_device_one);
    auto tensor_three = tensor_one * tensor_two;
  }

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
  std::cout << "Elapsed time: " << elapsed.count() << " ms\n";
}