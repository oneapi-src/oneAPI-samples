#ifndef __INTERSECTIONKERNEL_HPP__
#define __INTERSECTIONKERNEL_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// the kernel class names
// templated on the version of the kernel
// Best practice: forward declare the kernel names in the global scope
// to reduce compiler name mangling in the optimization reports.
template<int Version> class ProducerA;
template<int Version> class ProducerB;
template<int Version> class Worker;

// the pipe class names
// templated on the version of the kernel
template<int Version> class ProduceAPipeClass;
template<int Version> class ProduceBPipeClass;

//
// The base IntersectionKernel struct definition
// The idea of the IntersectionKernel struct is to define
// a templated function that we will (partially) override for different 
// versions of the kernel. This is called "partial template specialization".
// It is not possible to do partial template specialization on functions
// (the reason for why are outside the scope of this tutorial).
// However, one can do partial template specialization on classes and structs.
// Therefore, the IntersectionKernel struct will allow use to have a templated
// kernel that we will partially template on the 'Version' parameter to define
// different versions of the kernel.
//
template <int Version, int II, class APipe, class BPipe>
struct IntersectionKernel {
  int operator()(int a_size, int b_size) const;
};

//
// Version 0
// This is the baseline version of the algorithm with no optimizations
// Note that we have partially override this struct with Version=0
//
template <int II, class APipe, class BPipe>
struct IntersectionKernel<0, II, APipe, BPipe> {
  int operator()(int a_size, int b_size) const {
    // initialize variables
    unsigned int a = APipe::read();
    unsigned int b = BPipe::read();
    int a_count = 1;
    int b_count = 1;
    int n = 0;

    [[intel::initiation_interval(II)]]
    while (a_count < a_size || b_count < b_size) {
      // increment the intersection counter if the table elements match
      if (a == b) {
        n++;
      }

      ///////////////////////////////////////////////////////////////////////
      // To achieve an II of 1, all of the code in this block must occur in
      // in the same clock cycle. Why? The path taken by the NEXT iteration
      // of this loop depends on the result of this computation for the
      // CURRENT iteration of this loop. That is, the critical path is
      // two COMPAREs (a < b, a_count < a_size), an AND, an ADD (a_count++)
      // and a pipe read (::read()). This will results in a long critical
      // path and therefore either an increased II or decreased Fmax.
      if (a < b && a_count < a_size) {
        a = APipe::read();
        a_count++;
      } else if (b_count < b_size) {
        b = BPipe::read();
        b_count++;
      }
      ///////////////////////////////////////////////////////////////////////
    };

    // check the last elements
    if (a == b) {
      n++;
    }

    return n;
  }
};

//
// Version 1
// Note that we have partially override this struct with Version=1
//
template <int II, class APipe, class BPipe>
struct IntersectionKernel<1, II, APipe, BPipe> {
  int operator()(int a_size, int b_size) const  {
    // initialize variables
    unsigned int a = APipe::read();
    unsigned int b = BPipe::read();
    int a_count = 1;
    int b_count = 1;
    int a_count_next = 2;
    int b_count_next = 2;
    int n = 0;

    [[intel::initiation_interval(II)]]
    while (a_count < a_size || b_count < b_size) {
      // increment the intersection counter if the table elements match
      if (a == b) {
        n++;
      }

      ///////////////////////////////////////////////////////////////////////
      // In this version of the kernel, we remove the ADD from the critical
      // path by precomputing it. a_count_next stores the value for
      // next loop iteration that wants to increment a_count (and likewise
      // for (b_count_next and b_count). This removes the ADD from the 
      // critical path and replaces it by a read to the register holding
      // a_count_next. This results in a reduction to the critical path and
      // improved Fmax/II.
      if (a < b && a_count < a_size) {
        a = APipe::read();

        a_count = a_count_next;
        a_count_next++;
      } else if (b_count < b_size) {
        b = BPipe::read();

        b_count = b_count_next;
        b_count_next++;
      }
      ///////////////////////////////////////////////////////////////////////
    };

    // check the last elements
    if (a == b) {
      n++;
    }

    return n;
  }
};

//
// Version 2
// Note that we have partially override this struct with Version=2
//
template <int II, class APipe, class BPipe>
struct IntersectionKernel<2, II, APipe, BPipe> {
  int operator()(int a_size, int b_size) const {
    // initialize variables
    unsigned int a = APipe::read();
    unsigned int b = BPipe::read();
    int a_count = 1;
    int b_count = 1;
    int n = 0;

    bool a_count_inrange = a_count < a_size;
    bool b_count_inrange = b_count < b_size;
    bool a_count_next_inrange = a_count < (a_size - 1);
    bool b_count_next_inrange = b_count < (b_size - 1);
    bool keep_going = true;

    [[intel::initiation_interval(II)]]
    while (keep_going) {
      // increment the intersection counter if the table elements match
      if (a == b) {
        n++;
      }

      ///////////////////////////////////////////////////////////////////////
      // In this version of the kernel, we do the same optimization as
      // Version 1 for a_count by adding the variable a_count_next_next.
      // This precomputes the a_count values for the next TWO iterations
      // of the loop that read from APipe. We also precompute the check
      // for whether a_count and a_count_next are still in range using
      // the a_count_inrange and a_count_next_inrange variables.
      if (a < b && a_count_inrange) {
        a = APipe::read();

        // first update the variables that determine whether
        // the current counters are in range of the table
        a_count_inrange = a_count_next_inrange;
        a_count_next_inrange = (a_count < (a_size - 2));
        a_count++;
      } else if (b_count_inrange) {
        b = BPipe::read();

        // first update the variables that determine whether
        // the current counters are in range of the table
        b_count_inrange = b_count_next_inrange;
        b_count_next_inrange = (b_count < (b_size - 2));
        b_count++;
      }
      ///////////////////////////////////////////////////////////////////////

      keep_going = a_count_inrange || b_count_inrange;
    };

    // check the last elements
    if (a == b) {
      n++;
    }

    return n;
  }
};

#endif  /* __INTERSECTIONKERNEL_HPP__ */
