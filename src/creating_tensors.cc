#include "creating_tensors.hh"

void CreatingTensors::print() {
  // First way
  auto ex_torch =
      torch::stack({torch::tensor({1, 2}), torch::tensor({3, 4})}).view({2, 2});
  std::cout << ex_torch << std::endl;

  // Second way
  auto ex_torch2 = torch::tensor({{1, 2}, {3, 4}});
  std::cout << ex_torch2 << std::endl;

  // Third way, from vector
  // NOTE: The vector should be one dimensional (flattened)
  // from_blob does not copy
  // https://discuss.pytorch.org/t/can-i-initialize-tensor-from-std-vector-in-libtorch/33236/4
  auto shape = torch::IntArrayRef({2, 2});
  std::vector<uint8_t> vec{1, 2, 3, 4};
  auto ex_torch3 =
      torch::from_blob((uint8_t *)vec.data(),
                       shape,  // you could just write {2, 2} instead of shape
                       torch::kUInt8);
  std::cout << ex_torch3 << std::endl;

  // Flattening arrays
  std::vector<std::vector<float>> vec2{{1, 2}, {3, 4}};

  // Flattening through accumulate
  auto flattened =
      std::accumulate(vec2.begin(), vec2.end(), decltype(vec2)::value_type{},
                      [](auto &x, auto &y) {
                        x.insert(x.end(), y.begin(), y.end());
                        return x;
                      });
  auto ex_torch4 =
      torch::from_blob((float *)flattened.data(), {2, 2}, torch::kFloat);
  std::cout << ex_torch4 << std::endl;

  // Add'l flattening ideas
  // https://www.techiedelight.com/flatten-a-vector-of-vectors-in-cpp/
  // ranges-v3 lib: https://github.com/ericniebler/range-v3

  // Creating an all-ones matrix a.k.a matrix of ones
  auto ones = torch::ones({2, 3}, torch::kInt16);
  std::cout << ones << std::endl;

  // Creating a null matrix a.k.a zero matrix
  auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCUDA);
  auto zeros = torch::zeros({4, 5}, options);
  std::cout << zeros << std::endl;

  // Random matrix
  auto rand = torch::rand({2, 3}, torch::kF16);
  std::cout << rand << std::endl;

  // arange, NOT ARRANGE, just with one r
  auto arange_var = torch::arange(6);
  std::cout << arange_var << std::endl;

  // linspace
  auto linspace_var = torch::linspace(-10, 10, 5);
  std::cout << linspace_var << std::endl;

  // Identity matrix (unit matrix)
  auto eye = torch::eye(3);
  std::cout << eye << std::endl;

  // Diagonal matrix
  auto diag = torch::diag(torch::tensor({1, 2, 3}));
  std::cout << diag << std::endl;

  // Full matrix - fill with a value
  auto full = torch::full({2, 3}, 5);
  std::cout << full << std::endl;
}
