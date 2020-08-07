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
  Particle *particles;

  int nPart;       // number of particles
  int nSteps;      // number of integration steps
  RealType tStep;  // time step of the simulation

  int sFreq;  // sample frequency

  RealType kEnergy;  // kinetic energy

  double totTime;   // total time of the simulation
  double totFlops;  // total number of flops

  void InitPos();
  void InitVel();
  void InitAcc();
  void InitMass();

  inline void set_npart(const int &N) { nPart = N; }
  inline int get_npart() const { return nPart; }

  inline void set_tstep(const RealType &dt) { tStep = dt; }
  inline RealType get_tstep() const { return tStep; }

  inline void set_nsteps(const int &n) { nSteps = n; }
  inline int get_nsteps() const { return nSteps; }

  inline void set_sfreq(const int &sf) { sFreq = sf; }
  inline int get_sfreq() const { return sFreq; }

  void PrintHeader();
};

#endif
