/* ==============================================================
 * Copyright (c) 2015 - 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* On-board LED Blink
 * Turns the on-board LED on for one second, then off for one second,
 * repeatedly.
 */

#include <unistd.h>

#include <iostream>
#include <mraa.hpp>
#include <string>

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
      gpioPin = 1;
      break;
    default:
      string unknownPlatformMessage =
          "This sample uses the MRAA/UPM library for I/O access, "
          "you are running it on an unrecognized platform. "
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

  // create the pin object
  Gpio* d_pin = new Gpio(gpioPin);
  if (d_pin == NULL) {
    consoleMessage("MRAA couldn't initialize GPIO, exiting.");
    return MRAA_ERROR_UNSPECIFIED;
  }

  // set the pin as output
  if (d_pin->dir(DIR_OUT) != SUCCESS) {
    consoleMessage("Can't set digital pin as output, exiting.");
    return MRAA_ERROR_UNSPECIFIED;
  }

  // loop forever toggling the on board LED every second
  for (;;) {
    if (d_pin->write(0) != SUCCESS) {
      consoleMessage("MRAA cannot write pin value!");
      return MRAA_ERROR_UNSPECIFIED;
    }
    sleep(1);
    if (d_pin->write(1) != SUCCESS) {
      consoleMessage("MRAA cannot write pin value!");
      return MRAA_ERROR_UNSPECIFIED;
    }
    sleep(1);
  }

  return SUCCESS;
}
