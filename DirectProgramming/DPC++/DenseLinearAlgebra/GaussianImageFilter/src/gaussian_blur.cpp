//============================================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// ===========================================================================

//****************************************************************************
// 
// Description:
// This advanced SYCL code example implements a Gaussian blur filter, blurring
// a JPG or PNG image from the command line. The original file is not modified.
// The output file is a PNG image. 
//
// Usage:
// The program blurs an image provided on the command line.
//
//*****************************************************************************

// SYCL or oneAPI toolkit headers:
#include <sycl/sycl.hpp>

// Third party headers:
#include <cmath>
#include <iostream>
// These public domain headers implement useful image reading and writing
// functions. Find in ${oneAPI}/dev-utilities/include
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// Forward declaration of this example's SYCL kernels
class KernelFillGaussian;
class KernelGaussian;

using namespace sycl;
using namespace std;

// Attempts to determine a good local size. The best way to *control* 
// performance is to choose the sizes. The method here is to choose the 
// largest number, leq 64, which is a power-of-two, and divides the global
// work size evenly. In this code, it might prove most optimal to pad the
// image along one dimension so that the local size could be 64, but this 
// introduces other complexities.
range< 2 > GetOptimalLocalRange( range< 2 > globalSize, device hw ) 
{
  range< 2 > optimalLocalSize{ 0, 0 };

  // 64 is a good local size on GPU-like devices, as each compute unit is
  // made of many smaller processors. On non-GPU devices, 4 is a common vector
  // width. 
  if( hw.is_gpu() ) 
  {
    optimalLocalSize = range< 2 >( 64, 1 );
  } 
  else 
  {
    optimalLocalSize = range< 2 >( 4, 1 );  
  }

  // Here, for each dimension, we make sure that it divides the global size
  // evenly. If it doesn't, we try the next lowest power of two. Eventually
  // it will reach one, if the global size has no power of two component.
  for( int i = 0; i < 2; ++i ) 
  {
    while( globalSize[ i ] % optimalLocalSize[ i ] != 0 ) 
    {
      optimalLocalSize[ i ] = optimalLocalSize[ i ] >> 1;
    }
  }

  return optimalLocalSize;
}

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
          std::cout << 
          "Queue handler caught asynchronous SYCL exception:\n" << 
          e.what() << std::endl;
        }
    }
};

// The Gaussian program
int main( int argc, char* argv[] ) 
{
  bool bProgramError = false;

  // Validate user input
  if( argc < 2 ) 
  {
    std::cout
        << "Please provide a JPEG or PNG image as an argument to this program."
        << std::endl;
  }

  // ********************
  // Input image handling
  // ********************
  // The image dimensions will be set by the library, as will the number of
  // channels. However, passing a number of channels will force the image
  // data to be returned in that format, regardless of what the original image
  // looked like. The header has a mapping from int values to types - 4 means
  // RGBA. 
  int inputWidth = 0;
  int inputHeight = 0;
  int inputChannels = 0;
  
  // Number of color channels RGBA// Project files:
  const int numChannels = 4;
  const char *pImageFileName = argv[ 1 ];
  unique_ptr< unsigned char [] > pInputImg( stbi_load( pImageFileName, 
                     &inputWidth, &inputHeight, &inputChannels, numChannels ) );
  if( pInputImg == nullptr ) 
  {
    bProgramError = true;
    std::cout << "Failed to load image file (is argv[1] a valid image file?)"
              << std::endl;
    exit(-1);
  }

  // RAII resource
  unique_ptr< unsigned char [] > pOutputImg( 
    new unsigned char[ inputWidth * inputHeight * numChannels ] );
         
  try
  {
    sycl::device hw = device( sycl::cpu_selector_v );
    queue myQueue( hw, exception_handler );
    
    // *******************************************
    // Create gaussian convolution matrix and fill
    // *******************************************
    const float pi = std::atan( 1 ) * 4;
    constexpr auto guasStdDev = 2;     
    constexpr auto guasDelta = 6; 
    const int guasMatrixRange = (guasDelta * guasStdDev);
    const float guasStdDevFactor = 2 * guasStdDev * guasStdDev;
    const float piFactor = guasStdDevFactor * pi;
    const int gaussianBlurRange = guasMatrixRange * guasMatrixRange;
    vector< float > gaussianBlurMatrix( gaussianBlurRange );

    // The nd_range contains the total work (as mentioned previously) as
    // well as the local work size (i.e. the number of threads in the local
    // group). Here, we attempt to find a range close to the device's
    // preferred size that also divides the global size neatly.
    auto optRange = GetOptimalLocalRange( 
      range< 2 >{ guasMatrixRange, guasMatrixRange }, myQueue.get_device() );
    const nd_range< 2 > gaussianBlurNDRange( 
      range< 2 >{ guasMatrixRange, guasMatrixRange }, optRange );
    buffer bufGaussian( gaussianBlurMatrix );

    // Enqueue KernelFillGaussian
    myQueue.submit( [&]( handler &cgh ) 
    {
      const auto ptrGBlur = 
        bufGaussian.get_access< access::mode::discard_write >( cgh );
      cgh.parallel_for< KernelFillGaussian >( gaussianBlurNDRange, 
      [=]( nd_item< 2 > item ) 
      {
        // Get the 2D x and y indicies
        const auto idX = item.get_global_id( 0 );
        const auto idY = item.get_global_id( 1 );
        const auto width = item.get_group_range( 0 ) * 
                           item.get_local_range( 0 );
        const auto index = idX * width + idY;
        const auto x = idX - guasDelta;
        const auto y = idY - guasDelta;
        float gausVallue = sycl::exp( -1.0f * (x*x + y*y) / guasStdDevFactor );
        gausVallue /= piFactor;
        ptrGBlur[ index ] = gausVallue;
      });
    });

    // ********************************************************
    // Using gaussian convolution matrix, blur the input image.
    // ********************************************************
    
    // Images need a void * pointing to the data, and enums describing the
    // type of the image (since a void * carries no type information). It
    // also needs a range which describes the image's dimensions.
    using co = sycl::image_channel_order;
    using ct = sycl::image_channel_type;
    // The image data has been returned us an unsigned char [], but due to 
    // OpenCL restrictions, we must use it as a void *.
    void *pInputData = (void *) pInputImg.get();
    void *pOutputData = (void *) pOutputImg.get();
    // This range represents the full amount of work to be done across the
    // image. We dispatch one thread per pixel.
    range< 2 > imgRange( inputWidth, inputHeight );
    image< 2 > imageIn( pInputData, co::rgba, ct::unorm_int8, imgRange );
    image< 2 > imageOut( pOutputData, co::rgba, ct::unorm_int8, imgRange );
    optRange = GetOptimalLocalRange( imgRange, myQueue.get_device() );
    auto myRange = nd_range< 2 >( imgRange, optRange );
    constexpr auto offset = guasDelta; 

    // Enqueue KernelGaussian
    // Because of the dependency on the gaussian convolution grid, the call
    // graph will automatically schedule this kernel to run after the
    // KernelFillGaussian is complete.
    myQueue.submit( [&]( handler &cgh ) 
    {
      // Images still require accessors, like buffers, except the target is
      // always access::target::image.
      accessor< float4, 2, access::mode::read, access::target::image > 
                accImgInPtr( imageIn, cgh );
      accessor< float4, 2, access::mode::discard_write, access::target::image > 
                accImgOutPtr( imageOut, cgh );
      const auto ptrGBlur = 
                bufGaussian.get_access< access::mode::read >( cgh );
      
      // The sampler is used to map user-provided co-ordinates to pixels in
      // the image.
      sampler smpl( coordinate_normalization_mode::unnormalized,
                    addressing_mode::none, filtering_mode::nearest );

      // Setting breakpoints in the kernel code does not present the normal 
      // step through code behavior. Instead a breakpoint event is occurring
      // on each thread being executed and so switches to the context of 
      // that thread. To step through the code of a single thread, use the 
      // Intel gdb-oneapi command 'set scheduler-locking step' or 'on' in the 
      // IDE's debug console prompt. As this is not the main thread, be sure
      // to revert this setting on returning to debug any host side code. 
      // Use the command 'set scheduler-locking replay' or 'off'.  
      cgh.parallel_for< KernelGaussian >( myRange, [=](nd_item< 2 > item) 
      {
         const auto idY = item.get_global_id( 1 );
         const auto idX = item.get_global_id( 0 );
         const auto outputCoords = int2( idX, idY );
         // A boundary is used so the convolution grid does not fall off the
         // sides of the image. Keep it simple, just copy those pixels at the
         // edges of the image.
         const int hitY1 = idY - offset;
         const int hitY2 = inputHeight - idY - offset;
         const int hitX1 = idX - offset;
         const int hitX2 = inputWidth - idX - offset;
         const bool bBoundryY = (hitY1 < 0) || (hitY2 < 0);
         const bool bBoundryX = (hitX1 < 0) || (hitX2 < 0);
         float4 newPixel = float4( 0.0f, 0.0f, 0.0f, 0.0f );
                             
         if( !(bBoundryX || bBoundryY) )
         {
          // Perform a convolution on a central pixel at idX idY
          for( int x = 0; x < guasMatrixRange; x++ ) 
          {
            for( int y = 0; y < guasMatrixRange; y++ ) 
            {
              const auto index = x * guasMatrixRange + y;
              const float value = ptrGBlur[ index ];
              const auto inputCoords = 
                int2( idX + x - offset, idY + y - offset );
              newPixel += accImgInPtr.read( inputCoords, smpl ) * value;
            }
          }
         }
         else
         {
            // Just duplicate the pixel at idX idY
            const auto inputCoords = int2( idX, idY );
            newPixel = accImgInPtr.read( inputCoords, smpl );
         }
         newPixel.w() = 1.0f;
         accImgOutPtr.write( outputCoords, newPixel );

      });
    });
    // The host/main thread is asked to wait here until all enqueued kernels 
    // have completed execution.
    myQueue.wait_and_throw();
  }
  // Synchronous errors are classical C++ exceptions
  catch( sycl::exception const &e )
  {

    bProgramError = true;
    cout << 
    "Wrap catch caught synchronous SYCL exception:\n" << e.what() << std::endl;
  }

  if( bProgramError )
  {
    std::cout << "Program failed." << std::endl;
    return -1;
  }

  // ****************************
  // Output the new blurred image
  // ****************************
  
  // Attempt to change the name from x.png or x.jpg to x-blurred.png. 
  // If the code cannot find a '.', it simply appends "-blurred" to the name.
  std::string outputFilePath;
  std::string inputName( argv[ 1 ] );
  auto pos = inputName.find_last_of( "." );
  if( pos == std::string::npos ) 
  {
    outputFilePath = inputName + "-blurred";
  } 
  else 
  {
    inputName.erase( pos, inputName.size() );
    outputFilePath = inputName + "-blurred" + ".png";
  }

  stbi_write_png( outputFilePath.c_str(), inputWidth, inputHeight, numChannels,
                  pOutputImg.get(), 0 );

  std::cout << 
    "Program success, the image is successfully blurred!" << std::endl;
  
  return 0;
}