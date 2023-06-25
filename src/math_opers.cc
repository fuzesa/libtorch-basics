#include "math_opers.hh"

void MathOpers::print() {
  auto torch1 = torch::ones({2, 3});
  std::cout << torch1 << std::endl;

  auto torch2 = torch::full({2, 3}, 3);
  std::cout << torch2 << std::endl;

  // Addition
  std::cout << torch1 + torch2 << std::endl;

  // Subtraction
  std::cout << torch1 - torch2 << std::endl;

  // Multiplication
  std::cout << torch1 * torch2 << std::endl;

  // Division
  std::cout << torch1 / torch2 << std::endl;

  // Can be done with separate function calls, not just operators
  auto torch3 = torch::add(torch1, torch2);
  std::cout << torch3 << std::endl;

  // In-place operations
  torch1.add_(torch2);
  std::cout << torch1 << std::endl;
}