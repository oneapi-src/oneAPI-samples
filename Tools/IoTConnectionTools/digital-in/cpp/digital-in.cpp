/* ==============================================================
 * Copyright (c) 2015 - 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Digital input
 * Read values from a gpio digital input pin.
 */

#include <unistd.h>

#include <iostream>
#include <mraa.hpp>

// Define the following if using a Grove Pi Shield
#define USING_GROVE_PI_SHIELD
using namespace std;
using namespace mraa;

// leave warning/error message in console and wait for user to press Enter
void consoleMessage(const string& str) {
  cerr << str << endl;
  sleep(10);
}

// check if running as root
void checkRoot(void) {
  int euid = geteuid();
  if (euid) {
    consoleMessage(
        "This project uses Mraa I/O operations that may require\n"
        "'root' privileges, but you are running as non - root user.\n"
        "Passwordless keys(RSA key pairs) are recommended \n"
        "to securely connect to your target with root privileges. \n"
        "See the project's Readme for more info.\n\n");
  }
  return;
}

// set pin values depending on the current board (platform)
void initPlatform(int& gpioPin) {
  // check which board we are running on
  Platform platform = getPlatformType();
  switch (platform) {
    case INTEL_UP2:
#ifdef USING_GROVE_PI_SHIELD
      gpioPin = 4 + 512;  // D4 Connector (512 offset needed for the shield)
      break;
#endif
    case INTEL_UP:
    case INTEL_EDISON_FAB_C:
    case INTEL_GALILEO_GEN2:
      break;
    case INTEL_MINNOWBOARD_MAX:  // Same for Minnowboard Turbot
      gpioPin = 104;
      break;
    case INTEL_JOULE_EXPANSION:
      gpioPin = 101;
      break;
    case IEI_TANK:
      gpioPin = 0;
      break;
    default:
      string unknownPlatformMessage =
          "This sample uses the MRAA/UPM library for I/O access, "
          "you are running it on an unrecognized platform.\n "
          "You may need to modify the MRAA/UPM initialization code to "
          "ensure it works properly on your platform.\n";
      consoleMessage(unknownPlatformMessage);
  }
  return;
}

int main() {
  // Check access permissions for the current user
  // Can be commented out for targets with user level I/O access enabled
  checkRoot();

  int gpioPin = 13;
  initPlatform(gpioPin);
#ifdef USING_GROVE_PI_SHIELD
  addSubplatform(GROVEPI, "0");
#endif

  // create a GPIO object from MRAA for the pin
  Gpio* d_pin = new Gpio(gpioPin);
  if (d_pin == NULL) {
    consoleMessage("Can't create mraa::Gpio object, exiting");
    return MRAA_ERROR_UNSPECIFIED;
  }

  // set the pin as input
  if (d_pin->dir(DIR_IN) != SUCCESS) {
    consoleMessage("Can't set digital pin as input, exiting");
    return MRAA_ERROR_UNSPECIFIED;
  }

  // loop forever printing the digital input value every second
  for (;;) {
    uint16_t pin_value = d_pin->read();
    // if incorrect value input
    if (pin_value == UINT16_MAX) {
      consoleMessage("MRAA cannot read pin value!");
      return MRAA_ERROR_INVALID_PARAMETER;
    }
    cout << "value " << pin_value << endl;
    sleep(1);
  }

  return SUCCESS;
}
