#ifndef BASE_CLASS_HH
#define BASE_CLASS_HH

#include <torch/torch.h>

#include <iostream>

class BaseClass {
  virtual void print() = 0;
};

#endif  // BASE_CLASS_HH