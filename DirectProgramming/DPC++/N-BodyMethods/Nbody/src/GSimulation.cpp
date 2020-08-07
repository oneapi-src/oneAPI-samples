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

#include "GSimulation.hpp"
#include "dpc_common.hpp"
using namespace sycl;
using namespace dpc_common;

GSimulation::GSimulation() {
  std::cout << "===============================" << "\n";
  std::cout << " Initialize Gravity Simulation" << "\n";
  set_npart(16000);
  set_nsteps(10);
  set_tstep(0.1);
  set_sfreq(1);
}

void GSimulation::SetNumberOfParticles(int N) { set_npart(N); }

void GSimulation::SetNumberOfSteps(int N) { set_nsteps(N); }

void GSimulation::InitPos() {
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
  }
}

void GSimulation::InitVel() {
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(-1.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles[i].vel[0] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[1] = unif_d(gen) * 1.0e-3f;
    particles[i].vel[2] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation::InitAcc() {
  for (int i = 0; i < get_npart(); ++i) {
    particles[i].acc[0] = 0.f;
    particles[i].acc[1] = 0.f;
    particles[i].acc[2] = 0.f;
  }
}

void GSimulation::InitMass() {
  RealType n = static_cast<RealType>(get_npart());
  std::random_device rd;  // random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<RealType> unif_d(0.0, 1.0);

  for (int i = 0; i < get_npart(); ++i) {
    particles[i].mass = n * unif_d(gen);
  }
}

void GSimulation::Start() {
  RealType dt = get_tstep();
  int n = get_npart();
  RealType* energy = new RealType[n];
  for (int i = 0; i < n; i++) energy[i] = 0.f;
  // allocate particles
  particles = new Particle[n];

  InitPos();
  InitVel();
  InitAcc();
  InitMass();

  PrintHeader();

  totTime = 0.;

  const float kSofteningSquared = 1e-3f;
  // prevents explosion in the case the particles are really close to each other
  const float kG = 6.67259e-11f;

  double nd = double(n);
  double gFlops = 1e-9 * ((11. + 18.) * nd * nd + nd * 19.);
  double av = 0.0, dev = 0.0;
  int nf = 0;
  auto R = range<1>(n);
  // Create a queue to the selected device and enabled asynchronous exception
  // handling for that queue
  queue q(default_selector{}, exception_handler);
  // Create SYCL buffer for the Particle array of size "n"
  buffer pbuf(particles, R, {cl::sycl::property::buffer::use_host_ptr()});
  // Create SYCL buffer for the ener array
  buffer ebuf(energy, R, {cl::sycl::property::buffer::use_host_ptr()});

  TimeInterval t0;
  int nSteps = get_nsteps();
  for (int s = 1; s <= nSteps; ++s) {
    TimeInterval ts0;
    q.submit([&](handler& h) {
       auto p = pbuf.get_access<access::mode::read_write>(h);
       h.parallel_for(R, [=](id<1> i) {
         RealType acc0 = p[i].acc[0];
         RealType acc1 = p[i].acc[1];
         RealType acc2 = p[i].acc[2];
         for (int j = 0; j < n; j++) {
           RealType dx, dy, dz;
           RealType distanceSqr = 0.0f;
           RealType distanceInv = 0.0f;

           dx = p[j].pos[0] - p[i].pos[0];  // 1flop
           dy = p[j].pos[1] - p[i].pos[1];  // 1flop
           dz = p[j].pos[2] - p[i].pos[2];  // 1flop

           distanceSqr =
               dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
           distanceInv = 1.0f / sycl::sqrt(distanceSqr);         // 1div+1sqrt

           acc0 += dx * kG * p[j].mass * distanceInv * distanceInv *
                   distanceInv;  // 6flops
           acc1 += dy * kG * p[j].mass * distanceInv * distanceInv *
                   distanceInv;  // 6flops
           acc2 += dz * kG * p[j].mass * distanceInv * distanceInv *
                   distanceInv;  // 6flops
         }
         p[i].acc[0] = acc0;
         p[i].acc[1] = acc1;
         p[i].acc[2] = acc2;
       });
     })
        .wait_and_throw();
    q.submit([&](handler& h) {
       auto p = pbuf.get_access<access::mode::read_write>(h);
       auto e = ebuf.get_access<access::mode::read_write>(h);
       h.parallel_for(R, [=](id<1> i) {
         p[i].vel[0] += p[i].acc[0] * dt;  // 2flops
         p[i].vel[1] += p[i].acc[1] * dt;  // 2flops
         p[i].vel[2] += p[i].acc[2] * dt;  // 2flops

         p[i].pos[0] += p[i].vel[0] * dt;  // 2flops
         p[i].pos[1] += p[i].vel[1] * dt;  // 2flops
         p[i].pos[2] += p[i].vel[2] * dt;  // 2flops

         p[i].acc[0] = 0.f;
         p[i].acc[1] = 0.f;
         p[i].acc[2] = 0.f;

         e[i] = p[i].mass *
                (p[i].vel[0] * p[i].vel[0] + p[i].vel[1] * p[i].vel[1] +
                 p[i].vel[2] * p[i].vel[2]);  // 7flops
       });
     })
        .wait_and_throw();
    q.submit([&](handler& h) {
       auto e = ebuf.get_access<access::mode::read_write>(h);
       h.single_task([=]() {
         for (int i = 1; i < n; i++) e[0] += e[i];
       });
     })
        .wait_and_throw();
    auto a = ebuf.get_access<access::mode::read_write>();
    kEnergy = 0.5 * a[0];
    a[0] = 0;

    double elapsedSeconds = ts0.Elapsed();
    if (!(s % get_sfreq())) {
      nf += 1;
      std::cout << " " << std::left << std::setw(8) << s << std::left
                << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(5) << std::setw(12) << kEnergy
                << std::left << std::setprecision(5) << std::setw(12)
                << elapsedSeconds << std::left << std::setprecision(5)
                << std::setw(12) << gFlops * get_sfreq() / elapsedSeconds
                << "\n";
      if (nf > 2) {
        av += gFlops * get_sfreq() / elapsedSeconds;
        dev += gFlops * get_sfreq() * gFlops * get_sfreq() /
               (elapsedSeconds * elapsedSeconds);
      }
    }

  }  // end of the time step loop
  totTime = t0.Elapsed();
  totFlops = gFlops * get_nsteps();

  av /= (double)(nf - 2);
  dev = sqrt(dev / (double)(nf - 2) - av * av);

  std::cout << "\n";
  std::cout << "# Total Time (s)     : " << totTime << "\n";
  std::cout << "# Average Performance : " << av << " +- " << dev << "\n";
  std::cout << "===============================" << "\n";
}

void GSimulation::PrintHeader() {
  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << "\n";

  std::cout << "------------------------------------------------" << "\n";
  std::cout << " " << std::left << std::setw(8) << "s" << std::left
            << std::setw(8) << "dt" << std::left << std::setw(12) << "kenergy"
            << std::left << std::setw(12) << "time (s)" << std::left
            << std::setw(12) << "GFlops" << "\n";
  std::cout << "------------------------------------------------" << "\n";
}

GSimulation::~GSimulation() { delete particles; }
