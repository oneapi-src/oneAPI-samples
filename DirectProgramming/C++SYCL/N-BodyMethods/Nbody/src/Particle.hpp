#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include <cmath>

#include "type.hpp"

struct Particle {
 public:
  Particle() : pos{}, vel{}, acc{}, mass{} {};
  RealType pos[3];
  RealType vel[3];
  RealType acc[3];
  RealType mass;
};

#endif
