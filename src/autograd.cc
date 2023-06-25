#include "autograd.hh"

void AutoGrad::print() {
  const torch::Device vid_card0(torch::kCUDA, 0);
  const auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(vid_card0)
                           .requires_grad(true);
  const auto x = torch::tensor({5}, options);
  const auto y = torch::tensor({6}, options);
  std::cout << "x: " << x << std::endl;
  std::cout << "y: " << y << std::endl;

  const auto z = ((x * x) * y) + (x * y);
  std::cout << "z: " << z << std::endl;

  // Convert to scalar
  const auto s = z.sum();
  std::cout << "s: " << s << std::endl;

  // Compute gradient
  s.backward();
  std::cout << "x.grad(): " << x.grad() << std::endl;
  std::cout << "y.grad(): " << y.grad() << std::endl;

  // Build intuition
  const auto x2 = torch::rand({10}, vid_card0);
  const auto y2 = 1.8 * x2 + 32;
  std::cout << "x2: " << x2 << std::endl;
  std::cout << "y2: " << y2 << std::endl;

  const auto w = torch::ones(
      {1}, torch::TensorOptions().requires_grad(true).device(vid_card0));
  const auto b = torch::ones(
      {1}, torch::TensorOptions().requires_grad(true).device(vid_card0));

  // Predicted values of the output
  const auto y_hat = w * x2 + b;

  // Loss function
  const auto loss = ((y_hat - y2) * (y_hat - y2)).sum();
  std::cout << "loss: " << loss << std::endl;

  // Compute gradient
  loss.backward();
  std::cout << "w.grad(): " << w.grad() << std::endl;
  std::cout << "b.grad(): " << b.grad() << std::endl;

  // Another go with more data
  auto x3 = torch::randint(
      -100, 100, {100},
      torch::TensorOptions().dtype(torch::kF32).device(vid_card0));

  auto y3 = 1.8 * x3 + 32;
  auto w2 = torch::ones(
      {1}, torch::TensorOptions().requires_grad(true).device(vid_card0));
  auto b2 = torch::ones(
      {1}, torch::TensorOptions().requires_grad(true).device(vid_card0));
  auto y_hat2 = (w2 * x3) + b2;
  auto epochs = 100000;
  auto learning_rate = 0.000001;

  for (auto epoch = 0; epoch < epochs; ++epoch) {
    auto loss = (y_hat2 - y3).pow(2).sum();
    loss.backward();

    // Turn off gradient tracking,
    // so it won't be considered a relationship
    {
      torch::NoGradGuard no_grad;
      w2 -= learning_rate * w2.grad();
      b2 -= learning_rate * b2.grad();

      // Reset gradients
      w2.mutable_grad().zero_();
      b2.mutable_grad().zero_();
    }

    y_hat2 = (w2 * x3) + b2;
  }

  // Results should be close to 1.8 and 32
  std::cout << "w2: " << w2.item() << std::endl;
  std::cout << "b2: " << b2.item() << std::endl;
}