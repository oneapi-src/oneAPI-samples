/* ==============================================================
 * Copyright (c) 2015 - 2020 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Analog input
 * Read values from a gpio analog input pin.
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
      gpioPin = 2 + 512;  // A2 Connector (512 offset needed for the shield)
      break;
#endif
    default:
      string unknownPlatformMessage =
          "This sample uses the MRAA/UPM library for I/O access, "
          "you are running it on an unrecognized platform.\n"
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

  int gpioPin = 2;
  initPlatform(gpioPin);

#ifdef USING_GROVE_PI_SHIELD
  addSubplatform(GROVEPI, "0");
#endif
  // create an analog input object from MRAA using the pin
  Aio* a_pin = new Aio(gpioPin);
  if (a_pin == NULL) {
    consoleMessage("Can't create mraa::Aio object, exiting");
    return MRAA_ERROR_UNSPECIFIED;
  }

  // loop forever printing the input value every second
  for (;;) {
    uint16_t pin_value;
    try {
      // read the current input voltage
      pin_value = a_pin->read();
    } catch (const invalid_argument& readExc) {
      // if incorrect voltage value input
      cerr << "Invalid argument, exception thrown: " << readExc.what() << endl;
      consoleMessage("MRAA cannot read pin value!");
      return MRAA_ERROR_INVALID_PARAMETER;
    }
    cout << "analog input value " << pin_value << endl;
    sleep(1);
  }

  return SUCCESS;
}
