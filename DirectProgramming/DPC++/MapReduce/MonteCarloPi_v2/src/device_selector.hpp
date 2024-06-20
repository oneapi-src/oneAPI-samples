//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

// SYCL or oneAPI toolkit headers:
#include <sycl/sycl.hpp>

// Third party headers:
#include <iostream>

using namespace std;
using namespace sycl;

//++
//============================================================================
// Details: Common code utility. User defined enumerate of the SYCL device 
//          target typesdesired.
//          An 'eHost' is admitted because SYCL 2020 depricates host device
//          selection.
//--
enum EDevsWeWant
{
  eNotValid = 0,      // Default
  eCPU = 1,
  eGPU = 2,
  eAccelerator = 3,   // i.e. a FPGA type device
  eCount = 4          //  Always the last one
};
  
//++
//============================================================================
// Details: Common code utility. User defined target device proxy.
//          After the utility has discoverd available devices on the system,
//          this structure holds/caches information about the device. 
//          Forms a proxy device object representing an actual possible target
//          device found on the system.
//--
struct SDeviceFoundProxy final
{
  EDevsWeWant   eDevice = eNotValid;    // The type of real device we want to
                                        // use to run kernels on. 
  string        strDeviceName = "";     // THe proxy label (ID) for a real 
                                        // device.
  bool          bAvailable = false;     // True = can be used, 
                                        // False = not found on the system.
  bool          bActiveTarget = false;  // True = use it, false = stop using.
  int           nScore = 0;             // User defined score of the device.
  sycl::device  theDevice;              // Copy of the real device found.
}; 

//++
//============================================================================
// Details: Common code utility. Rudimentry error reporting system. Used by
//          utility class to explicity aid the user or the programmer of any
//          issues that have occurred.
//--
struct FnResult final
{
  bool bSuccess = true;
  string strErrMsg = "";
};

//++
//============================================================================
// Details: Common code utility. A basic utility class to wrap up functions
//          that can discover, then target the acceleration devices found on 
//          a system. 
//
// Docs:  https://www.intel.com/content/www/us/en/developer/articles/
//        technical/device-discovery-with-sycl.html#gs.nhyd7s
//        https://registry.khronos.org/SYCL/specs/sycl-2020/html/
//        sycl-2020.html#sec:device-selection
//
// It can find all the available device targets on a system.
// But it will only store a list of the first device found of the following
// type and criteria: cpu, gpu and accelerator.
// Each device in the list can be set to be an active target.
// The last or latest call to function SetDevToActive() will change the 
// device proxy object returned by GetDevUsersFirstChoice() to be that device.  
// -- 
class CUtilDeviceTargets final
{
  // Definitions:
  public:
    typedef std::vector< SDeviceFoundProxy > ListDevicesFound_t;

  // Static method:
  public:
    static FnResult       DiscoverPlatformsDevicesAvailable( string &vrstrPlatformAndDevices );
    static const string&  GetInputOptionDiscoverDevice();
    static FnResult       GetQueuesCurrentDevice( const queue &vrQ, string &vrstr );
 
  // Methods:
  public:
    CUtilDeviceTargets();
    ~CUtilDeviceTargets();
 
    FnResult                  DiscoverDevsWeWant();
    const ListDevicesFound_t &GetListDevs() const;
    const SDeviceFoundProxy  *GetDevUsersFirstChoice() const;
    FnResult                  SetDevToActive( const string &rvDeviceName, const bool vbActive );
  
  // Attributes:
  private:
    ListDevicesFound_t  m_listDeviceTargets;  
    static string       m_strDiscoverDeviceInputOption;
    SDeviceFoundProxy  *m_pDeviceUserFirstChoice;       // NULL = a choice has not been made
};

// Instantiations:
string CUtilDeviceTargets::m_strDiscoverDeviceInputOption = "discover_devices";

//++
// Details: CUtilDeviceTargets constructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CUtilDeviceTargets::CUtilDeviceTargets()
: m_pDeviceUserFirstChoice( nullptr ) 
{}

//++
// Details: CUtilDeviceTargets destructor.
// Type:    Method.
// Args:    None.
// Return:  None.
// Throws:  None.
//--
CUtilDeviceTargets::~CUtilDeviceTargets()
{
  // Release
  m_pDeviceUserFirstChoice = nullptr;
}

//++
// Details:  Return a report on the specified SYCL queue stating its current
//           real target device and the device platform.
//
// Type:    Method.
// Args:    vrQ - (R) The queue to query.
//          vrstrReport - (W) The report text. 
// Return:  FnResult - Status of the function's operational success.
// Throws:  SYCL implemenation may throw.
//--
FnResult CUtilDeviceTargets::GetQueuesCurrentDevice( const queue &vrQ, string &vrstrReport )
{
  FnResult status;

  vrstrReport = "[SYCL] Using device: [";
  vrstrReport += vrQ.get_device().get_info< info::device::name >();
  vrstrReport += "] from [";
  vrstrReport += vrQ.get_device().get_platform().get_info< info::platform::name >();
  vrstrReport += "]";

  return status;
}

//++
// Details: Returns the program's text label for the user input option
//          to choose to discover all the available device on a system.
// Type:    Method.
// Args:    None.
// Return:  string& - Text label.
// Throws:  None.
//--
const string& CUtilDeviceTargets::GetInputOptionDiscoverDevice()
{
  return m_strDiscoverDeviceInputOption;
}

//++
// Details: Returns the pointer to the proxy object in the list of 
//          discovery device proxies.
// Type:    Method.
// Args:    None.
// Return:  SDeviceFoundProxy* - pointer to object.
// Throws:  None.
//--
const SDeviceFoundProxy * CUtilDeviceTargets::GetDevUsersFirstChoice() const
{
  return m_pDeviceUserFirstChoice;
}

//++
// Details:  A pointer to the current user's choice of target is made on this
//           function being successful. 
//
// If this function fails, the pointer retains the last or
// remains NULL (a choice was never made).
//
// Type:    Method.
// Args:    string &rvDeviceName - (R) Proxy's label or ID text.
//          bool vbActive - (R) True = Use the device, False = disable use. 
// Return:  FnResult - Status of the function's operational success.
// Throws:  None.
//--
FnResult CUtilDeviceTargets::SetDevToActive( const string &rvDeviceName, const bool vbActive )
{
  FnResult status;

  bool bFoundDevice = false;
  for( SDeviceFoundProxy &rDev : m_listDeviceTargets )
  {
      if( rDev.strDeviceName == rvDeviceName )
      {
        bFoundDevice = true;
        rDev.bActiveTarget = vbActive;
        m_pDeviceUserFirstChoice = &rDev;
        break;
      }
  }
  if( !bFoundDevice )
  {
    status.bSuccess = false;
    status.strErrMsg = "Device '" + rvDeviceName;
    status.strErrMsg += "' not found in list of available device targets";
  }

  return status;
}
 
//++
// Details:  Discovers all the SYCL target devices available and assigns them
//           to a target device proxy object. A device proxy object holds
//           the criteria for a real device. All proxies created are
//           disabled until a real device is found to match it. A programmer
//           has to still set the proxy device as active to target that 
//           device it represent.
//
// Call this function before at the earliest opportunity and 
// before other functions in this class as it makes a list
// of the target devices we are aiming to use.
//
// A limitation of this function it will only assign the first real device that
// matches the proxy criteria. Any subsequent same or similar devices are
// ignored.
//
// Type:    Method.
// Args:    None. 
// Return:  FnResult - Status of the function's operational success.
// Throws:  SYCL implemenation may throw.
//--
FnResult CUtilDeviceTargets::DiscoverDevsWeWant()
{ 
  FnResult status;

  SDeviceFoundProxy accelerator{ eAccelerator, "accelerator", false };
  SDeviceFoundProxy cpu{ eCPU, "cpu", false };
  SDeviceFoundProxy gpu{ eGPU, "gpu", false };
 
  for( const auto platform : platform::get_platforms() )
  {
    for( const auto device : platform.get_devices() )
    {
      // Get first available device of each type
      if( !accelerator.bAvailable && device.is_accelerator() )
      {
        accelerator.bAvailable = true;
        accelerator.theDevice = device;
      }
      else if( !cpu.bAvailable && device.is_cpu() )
      {
        cpu.bAvailable = true;
        cpu.theDevice = device;
      }
      else if( !gpu.bAvailable && device.is_gpu() )
      {
        gpu.bAvailable = true;
        gpu.theDevice = device;
      }
    }
  }

  m_listDeviceTargets.push_back( accelerator );
  m_listDeviceTargets.push_back( cpu );
  m_listDeviceTargets.push_back( gpu );

  return status;
}

//++
// Details: Returns the list of proxy device objects the programmer has 
//          defined and wants found on the system. Some proxy objects
//          may be set to not available (and inactive) if not matching
//          devices has been found on the system.
// Type:    Method.
// Args:    None. 
// Return:  ListDevicesFound_t - List of proxy device objects.
// Throws:  None.
//--
const CUtilDeviceTargets::ListDevicesFound_t & CUtilDeviceTargets::GetListDevs() const
{
  return m_listDeviceTargets;
}

//++
// Details: Prints to std out all the SYCL device targets discovered on the
//          wanted to be used.  
// Type:    Method.
// Args:    string& vrstrPlatformAndDevices - (W) A report of found devices. 
// Return:  FnResult - Status of the function's operational success.
// Throws:  SYCL implemenation may throw.
//--
FnResult CUtilDeviceTargets::DiscoverPlatformsDevicesAvailable( string &vrstrPlatformAndDevices )
{
  FnResult status;

  vrstrPlatformAndDevices = "";
  bool bFoundPlatforms = false;
  bool bFoundDevices = false;
  for( const auto platform : platform::get_platforms() )
  {
    bFoundPlatforms = true;
    vrstrPlatformAndDevices += "Platform: ";
    vrstrPlatformAndDevices += platform.get_info< info::platform::name >();
    vrstrPlatformAndDevices += "\n";

    for( const auto device : platform.get_devices() )
    {
      bFoundDevices = true;
      vrstrPlatformAndDevices += "\tDevice: ";
      vrstrPlatformAndDevices += device.get_info< info::device::name >();
      vrstrPlatformAndDevices += "\n";
    }
  }
  if( !bFoundPlatforms && !bFoundDevices )
  {
    vrstrPlatformAndDevices = "No SYCL targeted platforms or devices found.";
  }

  return status;
}

//++
// Details: Checks the user's input is valid. If not a help message if formed
//          and returned. If valid, the matching proxy device object
//          discovered earlier is made active for use by the program.  
// Type:    Function.
// Args:    vrDevList- (RW) Utililty object managing proxy device objects.
//          argc - (R) Program's input arguments count. 
//          argv - (R) Program's list of input arguments.
// Return:  FnResult - Status of the function's operational success.
// Throws:  None.
//--
FnResult UserCheckTheirInput( CUtilDeviceTargets &vrDevList, int argc, char* argv[] ) 
{
  FnResult status;

  const CUtilDeviceTargets::ListDevicesFound_t &rDevs = vrDevList.GetListDevs();
  string strListDevsOptionsToUser;
  for( const SDeviceFoundProxy d : rDevs )
  {
    strListDevsOptionsToUser += d.strDeviceName + "|";
  }
  strListDevsOptionsToUser += CUtilDeviceTargets::GetInputOptionDiscoverDevice();

  if( argc < 2 ) 
  {
    status.bSuccess = false;
    status.strErrMsg = "Usage: " + string( argv[ 0 ] ) + " <";
    status.strErrMsg += strListDevsOptionsToUser;
    status.strErrMsg += ">";
    return status;
  }

  bool bTargetDevMatch = false;
  const string strArg{ argv[ 1 ] };
  for( const SDeviceFoundProxy d : rDevs )
  {
    if( strArg == d.strDeviceName )
    {
      bTargetDevMatch = true;
      status = vrDevList.SetDevToActive( strArg, true );
      break;
    }
  }
  if( status.bSuccess && !bTargetDevMatch && 
     (strArg != CUtilDeviceTargets::GetInputOptionDiscoverDevice() ) )
  {
    status.bSuccess = false;
    status.strErrMsg = "The device type cannot be found. Please enter a device type name from the list: ";
    status.strErrMsg += strListDevsOptionsToUser;
  }

  return status;
}

//++
// Details: Checks the user's input is the option to 'discover device target'
//          on the system.  
// Type:    Function.
// Args:    argv - (R) Program's list of input arguments.
//          bool rbDoDiscovery - (W) True = yes, the discovery option choosen.
// Return:  FnResult - Status of the function's operational success.
// Throws:  SYCL implemenation may throw.
//--
FnResult UserWantsToDiscoverPossibleTargets( char* argv[], bool &rbDoDiscovery )
{
  FnResult status;

  rbDoDiscovery = false;
  const string strArg{ argv[ 1 ] };
  if( strArg == CUtilDeviceTargets::GetInputOptionDiscoverDevice() )
  {
    string strPlatformAndDevicesReport;
    status = CUtilDeviceTargets::DiscoverPlatformsDevicesAvailable( strPlatformAndDevicesReport );
    if( status.bSuccess )
    {
      rbDoDiscovery = true;
      cout << strPlatformAndDevicesReport << std::endl;
    }
  }

  return status;
}