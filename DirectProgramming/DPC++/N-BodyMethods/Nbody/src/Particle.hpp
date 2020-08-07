#ifndef _PARTICLE_HPP
#define _PARTICLE_HPP
#include <cmath>
#include "type.hpp"

struct Particle {
 public:
  Particle() { Init(); }
  void Init() {
    pos[0] = 0.;
    pos[1] = 0.;
    pos[2] = 0.;
    vel[0] = 0.;
    vel[1] = 0.;
    vel[2] = 0.;
    acc[0] = 0.;
    acc[1] = 0.;
    acc[2] = 0.;
    mass = 0.;
  }
  RealType pos[3];
  RealType vel[3];
  RealType acc[3];
  RealType mass;
};

#endif
