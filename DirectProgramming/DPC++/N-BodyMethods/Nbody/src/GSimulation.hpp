//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// =============================================================
// Copyright (c) 2019 Fabio Baruffa
// Source: https://github.com/fbaru-dev/particle-sim
// MIT License
// =============================================================

#ifndef _GSIMULATION_HPP
#define _GSIMULATION_HPP

#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <CL/sycl.hpp>
#include "Particle.hpp"

class GSimulation {
 public:
  GSimulation();
  ~GSimulation();

  void Init();
  void SetNumberOfParticles(int N);
  void SetNumberOfSteps(int N);
  void Start();

 private:
  Particle *particles_;

  int npart_;       // number of particles
  int nsteps_;      // number of integration steps
  RealType tstep_;  // time step of the simulation

  int sfreq_;  // sample frequency

  RealType kenergy_;  // kinetic energy

  double total_time_;   // total time of the simulation
  double total_flops_;  // total number of flops

  void InitPos();
  void InitVel();
  void InitAcc();
  void InitMass();

  inline void set_npart(const int &N) { npart_ = N; }
  inline int get_npart() const { return npart_; }

  inline void set_tstep(const RealType &dt) { tstep_ = dt; }
  inline RealType get_tstep() const { return tstep_; }

  inline void set_nsteps(const int &n) { nsteps_ = n; }
  inline int get_nsteps() const { return nsteps_; }

  inline void set_sfreq(const int &sf) { sfreq_ = sf; }
  inline int get_sfreq() const { return sfreq_; }

  void PrintHeader();
};

#endif
