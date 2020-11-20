/*******************************************************************************
 * Copyright (c) 2018 IBM Corp.
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * and Eclipse Distribution License v1.0 which accompany this distribution.
 *
 * The Eclipse Public License is available at
 *    http://www.eclipse.org/legal/epl-v10.html
 * and the Eclipse Distribution License is available at
 *   http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * Contributors:
 *    Ranjan Dasgupta - Initial drop of deviceSample.c
 * 
 *******************************************************************************/

/*
 * This sample shows how-to develop a device code using Watson IoT Platform
 * iot-c device client library, connect and interact with Watson IoT Platform Service.
 * 
 * This sample includes the function/code snippets to perform the following actions:
 * - Initiliaze client library
 * - Configure device from configuration parameters specified in a configuration file
 * - Set client logging
 * - Enable error handling routines
 * - Send device events to WIoTP service
 * - Receive and process commands from WIoTP service
 *
 * SYNTAX:
 * deviceSample --config <config_file_path>
 *
 * Pre-requisite:
 * 1. This sample requires a device configuration file. 
 *    Refer to "Device Confioguration File" section of iot-c docs for details.
 * 2. Register type and id (specified in the configuration file) with IBM Watson IoT Platform service.
 *
 */

#include <stdio.h>
#include <signal.h>
#include <memory.h>
#include <stdlib.h>
#include <unistd.h>

/* Include header file of IBM Watson IoT platform C Client for devices */ 
#include "iotp_device.h"

char *configFilePath = NULL;
volatile int interrupt = 0;
char *progname = "deviceSample";
int useEnv = 0;
int testCycle = 0;

/* Usage text */
void usage(void) {
    fprintf(stderr, "Usage: %s --config config_file_path\n", progname);
    exit(1);
}

/* Signal handler - to support CTRL-C to quit */
void sigHandler(int signo) {
    signal(SIGINT, NULL);
    fprintf(stdout, "Received signal: %d\n", signo);
    interrupt = 1;
}

/* Get and process command line options */
void getopts(int argc, char** argv)
{
    int count = 1;

    while (count < argc)
    {
        if (strcmp(argv[count], "--config") == 0)
        {
            if (++count < argc)
                configFilePath = argv[count];
            else
                usage();
        }
        if (strcmp(argv[count], "--useEnv") == 0) {
            useEnv = 1;
        }
        if (strcmp(argv[count], "--testCycle") == 0) {
            if (++count < argc)
                testCycle = atoi(argv[count]);
            else
                usage();
        }
        count++;
    }
}

/* 
 * Device command callback function
 * Device developers can customize this function based on their use case
 * to handle device commands sent by WIoTP.
 * Set this callback function using API setCommandHandler().
 */
void  deviceCommandCallback (char* type, char* id, char* commandName, char *format, void* payload, size_t payloadSize)
{
    fprintf(stdout, "Received device command:\n");
    fprintf(stdout, "Type=%s ID=%s CommandName=%s Format=%s Len=%d\n", 
        type?type:"", id?id:"", commandName?commandName:"", format?format:"", (int)payloadSize);
    fprintf(stdout, "Payload: %s\n", payload?(char *)payload:"");

    /* Device developers - add your custom code to process device commands */
}

void logCallback (int level, char * message)
{
    if ( level > 0 )
        fprintf(stdout, "%s\n", message? message:"NULL");
    fflush(stdout);
}

void MQTTTraceCallback (int level, char * message)
{
    if ( level > 0 )
        fprintf(stdout, "%s\n", message? message:"NULL");
    fflush(stdout);
}

/* Main program */
int main(int argc, char *argv[])
{
    int rc = 0;
    int cycle = 0;

    /* 
     * DEV_NOTES:
     * Specifiy variable for WIoT client object 
     */
    IoTPConfig *config = NULL;
    IoTPDevice *device = NULL;

    /* check for args */
    if ( argc < 2 )
        usage();

    /* Set signal handlers */
    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    /* get argument options */
    getopts(argc, argv);

    /* Set IoTP Client log handler */
    rc = IoTPConfig_setLogHandler(IoTPLog_FileDescriptor, stdout);
    if ( rc != 0 ) {
        fprintf(stderr, "WARN: Failed to set IoTP Client log handler: rc=%d\n", rc);
        exit(1);
    }

    /* Create IoTPConfig object using configuration options defined in the configuration file. */
    rc = IoTPConfig_create(&config, configFilePath);
    if ( rc != 0 ) {
        fprintf(stderr, "ERROR: Failed to initialize configuration: rc=%d\n", rc);
        exit(1);
    }

    /* read additional config from environment */
    if ( useEnv == 1 ) {
        IoTPConfig_readEnvironment(config);
    } 

    /* Create IoTPDevice object */
    rc = IoTPDevice_create(&device, config);
    if ( rc != 0 ) {
        fprintf(stderr, "ERROR: Failed to configure IoTP device: rc=%d\n", rc);
        exit(1);
    }

    /* Set MQTT Trace handler */
    rc = IoTPDevice_setMQTTLogHandler(device, &MQTTTraceCallback);
    if ( rc != 0 ) {
        fprintf(stderr, "WARN: Failed to set MQTT Trace handler: rc=%d\n", rc);
    }

    /* Invoke connection API IoTPDevice_connect() to connect to WIoTP. */
    rc = IoTPDevice_connect(device);
    if ( rc != 0 ) {
        fprintf(stderr, "ERROR: Failed to connect to Watson IoT Platform: rc=%d\n", rc);
        fprintf(stderr, "ERROR: Returned error reason: %s\n", IOTPRC_toString(rc));
        exit(1);
    }

    /*
     * Set device command callback using API IoTPDevice_setCommandsHandler().
     * Refer to deviceCommandCallback() function DEV_NOTES for details on
     * how to process device commands received from WIoTP.
     */
    IoTPDevice_setCommandsHandler(device, deviceCommandCallback);

    /*
     * Invoke device command subscription API IoTPDevice_subscribeToCommands().
     * The arguments for this API are commandName, format, QoS
     * If you want to subscribe to all commands of any format, set commandName and format to "+"
     */
    char *commandName = "+";
    char *format = "+";
    IoTPDevice_subscribeToCommands(device, commandName, format);


    /* Use IoTPDevice_sendEvent() API to send device events to Watson IoT Platform. */

    /* Sample event - this sample device will send this event once in every 10 seconds. */
    char *data = "{\"d\" : {\"SensorID\": \"Test\", \"Reading\": 7 }}";

    while(!interrupt)
    {
        fprintf(stdout, "Send status event\n");
        rc = IoTPDevice_sendEvent(device,"status", data, "json", QoS0, NULL);
        fprintf(stdout, "RC from publishEvent(): %d\n", rc);

        if ( testCycle > 0 ) {
            cycle += 1;
            if ( cycle >= testCycle ) {
                break;
            }
        }
        sleep(10);
    }

    fprintf(stdout, "Publish event cycle is complete.\n");

    /* Disconnect device */
    rc = IoTPDevice_disconnect(device);
    if ( rc != IOTPRC_SUCCESS ) {
        fprintf(stderr, "ERROR: Failed to disconnect from  Watson IoT Platform: rc=%d\n", rc);
        exit(1);
    }

    /* Destroy client */
    IoTPDevice_destroy(device);

    /* Clear configuration */
    IoTPConfig_clear(config);

    return 0;

}

