/*Copyright 2020 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <iostream>

#include "GSimulation.hpp"

int main(int argc, char** argv) {
#ifdef DEBUG
  char* env = std::getenv("SYCL_BE");
  std::cout << "[ENV] SYCL_BE = " << (env ? env : "<not set>") << "\n";
#endif
  int n;      // number of particles
  int nStep;  // number ot integration steps

  GSimulation sim;

  if (argc > 1) {
    n = atoi(argv[1]);
    sim.SetNumberOfParticles(n);
    if (argc == 3) {
      nStep = atoi(argv[2]);
      sim.SetNumberOfSteps(nStep);
    }
  }

  sim.Start();

  return 0;
}
