/*
 * Copyright (c) 2018 - 2020 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <mraa.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>

//#include "mraa/common.hpp"
//#include "mraa/led.hpp"

// Define min and max brightness
#define LED_ON  255
#define LED_OFF 0

// This boolean controls the main loop
volatile bool should_run = true;

// Interrupt signal handler function
void sig_handler(int signum)
{
    if (signum == SIGINT) {
        printf("Exiting...\n");
        should_run = false;
    }
}

// leave warning/error message in console and wait for user to press Enter
void consoleMessage(char* str)
{
    printf("%s\n", str);
    sleep(10);
}

// check if running as root
void checkRoot(void)
{
    int euid = geteuid();
    if (euid) {
        fprintf(stderr, "This project uses Mraa I/O operations that may require\n"
            "'root' privileges, but you are running as non - root user.\n"
            "Passwordless keys(RSA key pairs) are recommended \n"
            "to securely connect to your target with root privileges. \n"
            "See the project's Readme for more info.\n\n");
    }
    return;
}

// Main function
int main()
{
    //Check access permissions for the current user
    //Can be commented out for targets with user level I/O access enabled
    checkRoot();

    // Perform a basic platform and version check
    if (mraa_get_platform_type() != MRAA_UP2) {
        fprintf(stderr, "This example is meant for the UP Squared board.\n"
            "Running it on different platforms will likely require code changes!\n");
    }

    if (strcmp(mraa_get_version(), "v1.9.0") < 0) {
        fprintf(stderr,"You need MRAA version 1.9.0 or newer to run this sample!\n");
        return 1;
    }

    // Register handler for keyboard interrupts
    signal(SIGINT, sig_handler);

    // Define our LEDs
    const int led_no = 4;
    char led_names[4][7] = {"red", "green", "yellow", "blue"};
    mraa_led_context leds[led_no];

    // Initialize them
    for (int i = 0; i < led_no; ++i) {
        leds[i] = mraa_led_init_raw(led_names[i]);
    }

    // Blink them in order and pause in-between
    while (should_run) {
        for (int i = 0; i < led_no; i++) {
            mraa_led_set_brightness(leds[i],LED_ON);
            usleep(200000);
            mraa_led_set_brightness(leds[i],LED_OFF);
        }
        usleep(200000);
    }

    // Cleanup and exit
    for (int i = 0; i < led_no; i++) {
        free(leds[i]);
    }
    return 0;
}
