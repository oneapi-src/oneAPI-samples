//============================================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// ===========================================================================

//****************************************************************************
// 
// Description:
// Example of a parallel inclusive scan in SYCL. Based on the two-phase 
// exclusive scan algorithm paper by Guy E. Blelloch titled "Prefix Sums and 
// Their Applications", 1990.
//
// Usage:
// The program takes one argument: host / cpu / gpu / accelerator.
//
//*****************************************************************************

// SYCL / Intel oneAPI files:
#include <sycl/sycl.hpp>
#include "dpc_common.hpp"

// Third party files:
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// This project's files:
#include "device_selector.hpp"

using namespace sycl;
using namespace std;

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
          std::cout << "Queue handler caught asynchronous SYCL exception:\n" << e.what() << std::endl;
        }
    }
};

// Forward decleration of functions
template< typename T, typename OP >
void ParallelScan( sycl::buffer< T, 1 > &bufIn, sycl::queue &q );
int TestSum( sycl::queue &q );
int TestFactorial( sycl::queue &q );

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

  int retResultSum = 0;
  int retResultFactorial = 0;

  try
  {
    queue myQueue( pUsersChosenDevice->theDevice, exception_handler );

    string strTheDeviceBeingUsed;
    fnResult = CUtilDeviceTargets::GetQueuesCurrentDevice( myQueue, strTheDeviceBeingUsed );
    if( !fnResult.bSuccess )
    {
      cerr << fnResult.strErrMsg << "\n";
      exit( -1 );
    }
    cout << strTheDeviceBeingUsed << "\n";

    retResultSum = TestSum( myQueue );
    retResultFactorial =  (retResultSum == 0) && TestFactorial( myQueue );
  }
  catch( sycl::exception const &e ) 
  {
    cout << "Fail; SYCL synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }
  catch( std::exception const &e ) 
  {
    cout << "Fail; Runtime synchronous exception occurred: " << e.what() << "\n";
    return -1;
  }

  if( (retResultSum != 0) || (retResultFactorial != 0) ) 
  {
    return 1;
  }

  cout << "Results are correct." << std::endl;
  
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// The identity element for a given operation.
template< typename T, typename OP >
struct SIdentity {};

template< typename T >
struct SIdentity< T, std::plus< T > > 
{
  static constexpr T value = 0;
};

template< typename T >
struct SIdentity< T, std::multiplies< T > > 
{
  static constexpr T value = 1;
};

template< typename T >
struct SIdentity< T, std::logical_or< T > > 
{
  static constexpr T value = false;
};

template< typename T >
struct SIdentity< T, std::logical_and< T > > 
{
  static constexpr T value = true;
};

// Dummy struct to generate unique kernel name types
template< typename T, typename U, typename V >
struct SKernelNameType {};


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Performs an inclusive scan with the given associative binary operation `OP`
// on the data in the `bufIn` buffer. Runs in parallel on the provided 
// accelerated hardware queue. Modifies the input buffer to contain the 
// results of the scan.
// Input size has to be a power of two. If the size isn't so, the input can
// easily be padded to the nearest power of two with any values, and the scan
// on the meaningful part of the data will stay the same.
template< typename T, typename OP >
void ParallelScan( sycl::buffer< T, 1 > &bufIn, sycl::queue &q ) 
{
  // Retrieve the device associated with the given queue.
  const sycl::device dev = q.get_device();
  const bool bHwIsCpu = dev.is_cpu();

  // Check if local memory is available. On host no local memory is fine, since
  if( !bHwIsCpu && 
      (dev.get_info< sycl::info::device::local_mem_type >() ==
       sycl::info::local_mem_type::none) ) 
  {
    throw std::runtime_error( "Non host device does not have local memory." );
  }

  const size_t bufSize = bufIn.size();
  if( ((bufSize & (bufSize - 1)) != 0) || (bufSize == 0) ) 
  {
    throw std::runtime_error( 
      "Given input buffer size is not a power of two." );
  }
  
  // Check if there is enough global memory.
  const size_t globalMemSize = 
      dev.get_info< sycl::info::device::global_mem_size >();
  if( !bHwIsCpu && (bufSize > (globalMemSize * 0.5) ) ) 
  {
    throw std::runtime_error( 
      "Non host device input size exceeds device global memory size." );
  }

  // Obtain device limits.
  const size_t maxWgroupSize =
      dev.get_info< sycl::info::device::max_work_group_size >();
  const size_t localMemSize = 
      dev.get_info< sycl::info::device::local_mem_size >();

  // Find a work-group size that is guaranteed to fit in local memory and is
  // below the maximum work-group size of the device.
  const size_t wgroupSizeLim =
      sycl::min( maxWgroupSize, localMemSize / (2 * sizeof( T )) );

  // Every work-item processes two elements, so the work-group size has to
  // divide this number evenly. 
  const size_t halfInBufSize = bufSize * 0.5;

  // Find the largest power of two that divides half_in_size and is within the
  // device limit.
  size_t wgroupSize = 0;
  size_t pow = size_t( 1 ) << (sizeof( size_t ) * 8 - 1);
  for( ; pow > 0; pow >>= 1 ) 
  {
    if( (halfInBufSize / pow) * pow == 
      halfInBufSize && (pow <= wgroupSizeLim) ) 
    {
      wgroupSize = pow;
      break;
    }
  }
  if( wgroupSize == 0 ) 
  {
    throw std::runtime_error(
        "Could not find an appropriate work-group size for the given input." );
  }
  const size_t dblWgrpSize = wgroupSize * 2;

  q.submit( [&]( sycl::handler &cgh ) 
  {
    const auto ptrData = 
      bufIn.template get_access< sycl::access::mode::read_write >( cgh );

    // Using scratch/local memory (to a work group) for faster memory 
    // access to compute the results
    sycl::accessor< T, 1, sycl::access::mode::read_write, 
                          sycl::access::target::local > 
                          scratch( wgroupSize * 2, cgh);

    // Use dummy struct as the unique kernel name.
    cgh.parallel_for< SKernelNameType< T, OP, class CScanSegments > >(
        sycl::nd_range< 1 >( halfInBufSize, wgroupSize ),
        [=]( sycl::nd_item< 1 > item ) 
        {
          const size_t gid = item.get_global_linear_id();
          const size_t lid = item.get_local_linear_id();

          // Read data into local memory.
          scratch[ 2 * lid ] = ptrData[ 2 * gid ];
          scratch[ 2 * lid + 1 ] = ptrData[ 2 * gid + 1 ];

          // Preserve the second input element to add at the end.
          const auto secondInput = scratch[ 2 * lid + 1 ];

          // Perform partial reduction (up-sweep) on the data. The `off`
          // variable is 2 to the power of the current depth of the
          // reduction tree. In the paper, this corresponds to 2^d.
          for( size_t off = 1; off < (wgroupSize * 2); off *= 2 ) 
          {
            // Synchronize local memory to observe the previous writes.
            item.barrier( sycl::access::fence_space::local_space );

            const size_t i = lid * off * 2;
            if( i < dblWgrpSize ) 
            {
              const size_t index = i + (off * 2) - 1;
              scratch[ index ] =
                  OP{}( scratch[ index ], scratch[ i + off - 1 ] );
            }
          }

          // Clear the last element to the identity before down-sweeping.
          if( lid == 0 ) 
          {
            scratch[ dblWgrpSize - 1 ] = SIdentity< T, OP >::value;
          }

          // Perform down-sweep on the tree to compute the whole scan.
          // Again, `off` is 2^d. 
          for( size_t off = wgroupSize; off > 0; off >>= 1 ) 
          {
            item.barrier( sycl::access::fence_space::local_space );

            const size_t i = lid * off * 2;
            if( i < dblWgrpSize ) 
            {
              const size_t indexT = i + off - 1;
              const size_t indexU = i + (off * 2) - 1;
              const auto t = scratch[ indexT ];
              const auto u = scratch[ indexU ];
              scratch[ indexT ] = u;
              scratch[ indexU ] = OP{}( t, u );
            }
          }

          // Synchronize again to observe results.
          item.barrier( sycl::access::fence_space::local_space );

          // To return an inclusive rather than exclusive scan result, shift
          // each element left by 1 when writing back into global memory. If
          // we are the last work-item, also add on the final element. 
          const size_t indexL1 = 2 * lid + 1;
          const size_t indexL2 = 2 * lid + 2;
          const size_t indexG1 = 2 * gid;
          const size_t indexG2 = 2 * gid + 1;
          ptrData[ indexG1 ] = scratch[ indexL1 ];
          if( lid == wgroupSize - 1 ) 
          {
            ptrData[ indexG2 ] = OP{}( scratch[ indexL1 ], secondInput );
          } 
          else 
          {
            ptrData[ indexG2 ] = scratch[ indexL2 ];
          }
        }    // [=]( sycl::nd_item< 1 > item )
      );  // cgh.parallel_for< SKernelNameType< T, OP, class CScanSegments > >(
  }); // q.submit( [&]( sycl::handler &cgh )

  // At this point we have computed the inclusive scans of this many segments.
  const size_t nSegments = halfInBufSize / wgroupSize;

  if( nSegments == 1 ) 
  {
    // If all of the data is in one segment, we're done.
    return;
  }
  // Otherwise we have to propagate the scan results forward into later
  // segments.

  // Allocate space for one (last) element per segment.
  sycl::buffer< T, 1 > bufEndSegment{ sycl::range< 1 >( nSegments ) };

  // Store the elements in this space.
  q.submit( [&](sycl::handler &cgh ) 
  {
    const auto ptrScans = bufIn.template get_access< 
                                   sycl::access::mode::read >( cgh );
    const auto ptrElems = bufEndSegment.template get_access< 
                                   sycl::access::mode::discard_write >( cgh );

    cgh.parallel_for< SKernelNameType< T, OP, class CCopyEndSeg > >(
        sycl::range< 1 >( nSegments ), 
        [=]( sycl::item< 1 > item ) 
        {
          const size_t id = item.get_linear_id();
          // Offset into the last element of each segment.
          ptrElems[ item ] = ptrScans[ (id + 1) * 2 * wgroupSize - 1 ];
        });
  });

  // Recursively scan the array of last elements.
  ParallelScan< T, OP >( bufEndSegment, q );

  // Add the results of the scan to each segment.
  q.submit( [&]( sycl::handler &cgh ) 
  {
    const auto ptrEndSegScan = bufEndSegment.template get_access< 
                                      sycl::access::mode::read >( cgh );
    const auto ptrDataIn = bufIn.template get_access< 
                                      sycl::access::mode::read_write >( cgh );

    cgh.parallel_for< SKernelNameType< T, OP, class CAddEndSeg > >(
        // Work with one less work-group, since the first segment is correct.
        sycl::nd_range< 1 >( halfInBufSize - wgroupSize, wgroupSize ),
        [=](sycl::nd_item< 1 > item) 
        {
          const size_t grpLinId = item.get_group_linear_id();

          // Start with the second segment.
          const size_t glbIdOff = item.get_global_linear_id() + wgroupSize;

          // Each work-group adds the corresponding number in the
          // "last element scan" array to every element in the group's
          // segment.
          ptrDataIn[ glbIdOff * 2 ] = OP{}( ptrDataIn[ glbIdOff * 2 ], 
                                            ptrEndSegScan[ grpLinId ] );
          ptrDataIn[ glbIdOff * 2 + 1 ] = OP{}( ptrDataIn[ glbIdOff * 2 + 1 ], 
                                                ptrEndSegScan[ grpLinId ] );
        });
  });
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Tests the scan with an addition operation, which is its most common use.
// Returns 0 if successful, a nonzero value otherwise.
int TestSum( sycl::queue &q ) 
{
  constexpr size_t size = 64;

  // Initializes a vector of sequentially increasing values.
  std::vector< int32_t > arrayIn( size );
  std::iota( arrayIn.begin(), arrayIn.end(), 1 );

  // Compute the prefix sum using SYCL.
  std::vector< int32_t > arraySum( arrayIn.size() );
  
  {
    // Read from `arrayIn`, but write into `arraySum`.
    buffer< int32_t, 1 > bufArrayIn( sycl::range< 1 >( arrayIn.size() ) );
    bufArrayIn.set_final_data( arraySum.data() );

    q.submit( [&](sycl::handler &cgh) 
    {
      const auto acc = 
          bufArrayIn.get_access< sycl::access::mode::write >( cgh );
      cgh.copy( arrayIn.data(), acc );
    });

    ParallelScan< int32_t, std::plus< int32_t > >( bufArrayIn, q );
  }

  // Compute the same operation using the standard library.
  std::vector < int32_t > arrayTestSum( arrayIn.size() );
  std::partial_sum( arrayIn.begin(), arrayIn.end(), arrayTestSum.begin() );

  cout << "\nSYCL compute's sum results:\n";
  for( auto a : arraySum ) 
  {
    cout << a << " ";
  }
  cout << std::endl;

  // Check if the results are correct.
  const bool bEqual = 
    std::equal( arraySum.begin(), arraySum.end(), arrayTestSum.begin() );
  if( !bEqual ) 
  {
    cout << "SYCL sum computation incorrect!\n";
    cout << "std::partial_sum's results:\n";
        
    for( auto a : arrayTestSum ) 
    {
      cout << a << " ";
    }
    
    return 1;
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Tests the scan with a multiply operation, which is a sequence of factorials.
// Returns 0 if successful, a nonzero value otherwise.
int TestFactorial( sycl::queue &q ) 
{
  // Anything above this size overflows the int64_t type
  constexpr size_t size = 16;

  // Initializes a vector of sequentially increasing values.
  std::vector< int64_t > arrayIn( size );
  std::iota( arrayIn.begin(), arrayIn.end(), 1 ); 

  // Compute a sequence of factorials using SYCL.
  std::vector< int64_t > arrayFact( arrayIn.size() );
  {
    // Read from `arrayIn`, but write into `arrayFact`.
    sycl::buffer< int64_t, 1 > bufArrayIn( sycl::range< 1 >( arrayIn.size() ));
    bufArrayIn.set_final_data( arrayFact.data() );
    q.submit( [&](sycl::handler &cgh ) 
    {
      const auto acc = bufArrayIn.get_access< sycl::access::mode::write >( cgh );
      cgh.copy( arrayIn.data(), acc );
    });

    ParallelScan< int64_t, std::multiplies< int64_t > >( bufArrayIn, q );
  }

  // Compute the same operation using the standard library.
  std::vector< int64_t > arrayTestFact( arrayIn.size() );
  std::partial_sum( arrayIn.begin(), arrayIn.end(), arrayTestFact.begin(),
                    std::multiplies< int64_t >{} );

  cout << "\nSYCL compute's factorial results:\n";
  for( auto a : arrayFact ) 
  {
    cout << a << " ";
  }
  cout << std::endl;
    
  // Check if the results are correct.
  const bool bEqual = std::equal( arrayFact.begin(), arrayFact.end(), 
                                  arrayTestFact.begin() );
  if( !bEqual ) 
  {
    cout << "SYCL factorial computation incorrect!\n";
    cout << "std::partial_sum's results:\n";
    
    for( auto a : arrayTestFact ) 
    {
      cout << a << " ";
    }
    
    return 1;
  }

  return 0;
}