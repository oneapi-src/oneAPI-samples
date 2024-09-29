//============================================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// ===========================================================================

//****************************************************************************
// 
// Description:
// Example of Monte-Carlo Pi approximation algorithm in SYCL. Also,
// demonstrating how to query the maximum number of work-items in a
// work-group to check if a kernel can be executed with the initially
// desired work-group size.
//
// Usage:
// The program takes one argument: host / cpu / gpu / accelerator.
//
//*****************************************************************************

// SYCL or oneAPI toolkit headers:
#include <sycl/sycl.hpp>

// Third party headers:
#include <algorithm>
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

// In-house headers:
#include "device_selector.hpp"

using namespace std;
using namespace sycl;

// Forward declerations:
size_t GetBestWorkGroupSize( size_t work_group_size, 
                             const sycl::device &device,
                             const sycl::kernel &kernel );

// Monte-Carlo Pi SYCL C++ functor
class CMonteCarloPiKernel 
{
  template< typename dataT >
  using readGlobalAccessor = sycl::accessor< 
                                dataT, 1, 
                                sycl::access::mode::read,
                                sycl::access::target::global_buffer >;
  template < typename dataT >
  using writeGlobalAccessor = sycl::accessor< 
                                dataT, 1, 
                                sycl::access::mode::write,
                                sycl::access::target::global_buffer >;
  template< typename dataT >
  using readWriteLocalAccessor = sycl::accessor<
                                    dataT, 1, 
                                    sycl::access::mode::read_write,
                                    sycl::access::target::local >;
 public:
  CMonteCarloPiKernel( readGlobalAccessor< sycl::cl_float2 > ptrPoints,
                       writeGlobalAccessor< sycl::cl_int > ptrResults,
                       readWriteLocalAccessor< sycl::cl_int > ptrResultsLocal )
  : m_ptrPoints( ptrPoints ),
    m_ptrResults( ptrResults ),
    m_ptrResultsLocal( ptrResultsLocal )
  {}

  // Functor kernel using a 1D ND-range of work items
  void operator()( sycl::nd_item< 1 > item ) const 
  {
    // Setting breakpoints in the kernel code does not present the normal 
    // step through code behavior. Instead a breakpoint event is occurring
    // on each thread being executed and so switches to the context of 
    // that thread. To step through the code of a single thread, use the 
    // gdb-oneapi command 'set scheduler-locking step' or 'on' in the 
    // IDE's debug console prompt. As this is not the main thread, be sure
    // to revert this setting on returning to debug any host side code. 
    // Use the command 'set scheduler-locking replay' or 'off'.

    const size_t idGlobal = item.get_global_id( 0 );
    const size_t idLocal = item.get_local_id( 0 );
    const size_t localDim = item.get_local_range( 0 );
    const size_t idGroup = item.get_group( 0 );

    // Get the point to work on
    const sycl::float2 point = m_ptrPoints[ idGlobal ];

    // Calculate the length - built-in SYCL function
    // length: sqrt(point.x * point.x + point.y * point.y)
    const float len = sycl::length( point );

    // Result is either 1 or 0
    m_ptrResultsLocal[ idLocal ] = (len <= 1.0f) ? 1 : 0;

    // Wait for the entire work group to get here.
    item.barrier( sycl::access::fence_space::local_space );

    // If work item 0 in work group, sum local values
    if( idLocal == 0 ) 
    {
      int sum = 0;
      for( size_t i = 0; i < localDim; i++ ) 
      {
        if( m_ptrResultsLocal[ i ] == 1 ) 
        {
          ++sum;
        }
      }

      // Store the sum in global memory
      m_ptrResults[ idGroup ] = sum;
    }
  }

 private:
  readGlobalAccessor< sycl::cl_float2 >   m_ptrPoints;
  writeGlobalAccessor< sycl::cl_int >     m_ptrResults;
  readWriteLocalAccessor< sycl::cl_int >  m_ptrResultsLocal;
};


// Asynchronous errors hander, catch faults in asynchronously executed code
// inside a command group or a kernel. They can occur in a different stackframe, 
// asynchronous error cannot be propagated up the stack. 
// By default, they are considered 'lost'. The way in which we can retrieve them
// is by providing an error handler function.
auto exception_handler = []( sycl::exception_list exceptions ) 
{
    for( std::exception_ptr const &e : exceptions ) 
    {
        try 
        {
          std::rethrow_exception( e );
        } 
        catch( sycl::exception const &e ) 
        {
          std::cout << "Queue handler caught asynchronous SYCL exception:\n" 
          << e.what() << std::endl;
        }
    }
};

// The Monto Carlo Pi program
int main( int argc, char *argv[] ) 
{
  CUtilDeviceTargets utilsDev;
  FnResult fnResult = utilsDev.DiscoverDevsWeWant();
  if( !fnResult.bSuccess )
  {
    cerr << "Program failure: Unable to discover target devices on this platform.\n";
    exit( -1 );
  }

  fnResult = UserCheckTheirInput( utilsDev, argc, argv ); 
  if( !fnResult.bSuccess ) 
  {
    cerr << fnResult.strErrMsg << "\n";
    exit( 1 );
  }

  bool bDoDevDiscovery = false;
  fnResult = UserWantsToDiscoverPossibleTargets( argv, bDoDevDiscovery );
  if( !fnResult.bSuccess )
  {
    cerr << fnResult.strErrMsg << "\n";
    exit( -1 );
  }
  if( bDoDevDiscovery ) exit( 1 );

  const SDeviceFoundProxy *pUsersChosenDevice = utilsDev.GetDevUsersFirstChoice();
  if( pUsersChosenDevice == nullptr )
  {
    cerr << "Program failure: Did not create a valid target device object.\n";
    exit( -1 );
  }

  constexpr size_t iterations = 1 << 20;
  size_t workGroupSize = 1 << 10;

  // Container for the sum calculated per each work-group.
  std::vector< sycl::cl_int > arrayResults;

  // Generate random points on the host - one point for each work item (thread)
  std::vector< sycl::float2 > arrayPoints( iterations );
  // Fill up with (pseudo) random values in the range: [0, 1]
  std::random_device r;
  std::default_random_engine e( r() );
  std::uniform_real_distribution< float > dist;
  std::generate( arrayPoints.begin(), arrayPoints.end(),
                [&r, &e, &dist]() 
                { 
                  return sycl::float2( dist( e ), dist( e ) ); 
                });

  try 
  {
    // Create a SYCL queue
    queue queue( pUsersChosenDevice->theDevice, exception_handler );

    string strTheDeviceBeingUsed;
    fnResult = CUtilDeviceTargets::GetQueuesCurrentDevice( queue, strTheDeviceBeingUsed );
    if( !fnResult.bSuccess )
    {
      cerr << fnResult.strErrMsg << "\n";
      exit( -1 );
    }
    cout << strTheDeviceBeingUsed << "\n";

    // Get device and display information: name and platform
    const sycl::device hw = queue.get_device();
    cout << "Selected " << hw.get_info< sycl::info::device::name >()
         << " on platform "
         << hw.get_info< sycl::info::device::platform >()
              .get_info< sycl::info::platform::name >()
         << std::endl;

    // Force online compilation of all kernels in the hwCntext now,
    // unless already compiled for the device ahead-of-time.
    const auto hwContext = queue.get_context();
    const sycl::kernel_id kernelID = 
        sycl::get_kernel_id< CMonteCarloPiKernel >();
    const auto hwKernelBundle = 
        sycl::get_kernel_bundle< sycl::bundle_state::executable >( hwContext );
    const sycl::kernel kernel = hwKernelBundle.get_kernel( kernelID );
    
    // If the desired work-group size doesn't satisfy the device, define a
    // perfect/max work-group depending on the selected device and kernel
    // maximum size allowance.
    workGroupSize = GetBestWorkGroupSize( workGroupSize, hw, kernel );

    // Size of the total sums that are going to be stored in the results vector
    // is set based on the defined work-group size.
    arrayResults.resize( iterations / workGroupSize );

    // Allocate device memory
    sycl::buffer< sycl::cl_float2 > buffPoints( arrayPoints.data(),
                                                sycl::range<1>( iterations ) );
    sycl::buffer< sycl::cl_int > buffResults( arrayResults.data(), 
                              sycl::range< 1 >( iterations / workGroupSize ) );

    queue.submit( [&](sycl::handler& cgh) 
    {
      const size_t global_size = iterations;
      const size_t local_size = workGroupSize;

      // Get access to the data (points and results) on the device
      const auto ptrPoints =
          buffPoints.get_access<sycl::access::mode::read,
                                sycl::access::target::device>( cgh );
      const auto ptrResults = 
          buffResults.get_access< sycl::access::mode::write >( cgh );
      
      // Allocate local memory on the device (to compute results)
      const sycl::accessor< sycl::cl_int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local >
          ptrResultsLocal( sycl::range< 1 >( local_size ), cgh );

      // Run the kernel
      cgh.parallel_for(
          sycl::nd_range< 1 >( sycl::range< 1 >( global_size ),
                               sycl::range< 1 >( local_size ) ),
          CMonteCarloPiKernel( ptrPoints, ptrResults, ptrResultsLocal ) );
    });
  } 
  catch( const sycl::exception &e ) 
  {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  } 
  catch( const std::exception &e ) 
  {
    std::cerr << "C++ exception caught: " << e.what() << std::endl;
    return 2;
  }

  // Sum the results (auto copied back to host)
  int inCircle = 0;
  for( int &result : arrayResults ) 
  {
    inCircle += result;
  }

  // Calculate the final result of "pi"
  float pi = (4.0f * inCircle) / iterations;
  std::cout << "pi = " << pi << std::endl;

  return 0;
}


// A helper to define a "perfect" work-group size dependant on selected device
// and kernel maximum allowance.
size_t GetBestWorkGroupSize( const size_t workGroupSize,
                             const sycl::device &device,
                             const sycl::kernel &kernel ) 
{
  if( device.is_cpu() ) 
  {
    const size_t maxDeviceWorkGroupSize =
        device.get_info< sycl::info::device::max_work_group_size >();

    // Check if the desired work-group size will be allowed on the host device
    // and query the maximum possible size on that device in case the desired
    // one is more than the allowed.
    if( workGroupSize > maxDeviceWorkGroupSize ) 
    {
      cout << "Maximum work-group size for device "
           << device.get_info< sycl::info::device::name >() << ": "
           << maxDeviceWorkGroupSize << std::endl;
      
      return maxDeviceWorkGroupSize;
    }

    return workGroupSize;
  } 
  else 
  {
    const size_t maxKernelWorkGroupSize = kernel.get_info<
       sycl::info::kernel_device_specific::work_group_size >( device );

    // Verify if the kernel can be executed with our desired work-group size,
    // and if it can't use the maximum allowed kernel work-group size for the
    // selected device.
    if( workGroupSize > maxKernelWorkGroupSize ) 
    {
      cout << "Maximum work-group size for "
           << typeid( CMonteCarloPiKernel ).name() << " on device "
           << device.get_info<sycl::info::device::name>() << ": "
           << maxKernelWorkGroupSize << "\n";
      
      return maxKernelWorkGroupSize;
    }
    
    // Otherwise, the work-size will stay the originally desired one
    return workGroupSize;
  }
}

