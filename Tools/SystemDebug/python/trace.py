#!/usr/bin/env python3

'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''


'''
OneApi tracecli example.  Decode AET data to csv.
'''

import os
from intel.tracecli import TraceAPI
from intel.tracecli.session.sessionconfiguration import \
    init_session_configuration, get_session_configuration


def main(npk_root: str):
    '''
    Simple main to decode an trace binary containing AET data using the
    TraceCLI.
    :param npk_root: Full path to an NPK_ROOT directory
    '''

    # Path to trace bin file
    file_bin = f"{npk_root}/examples/input/mipi_aet_fake_trace.bin"

    # Initialize the session config
    init_session_configuration(npk_root)
    session_config = get_session_configuration()

    # Set the PVSS
    pvss = "CMP:H:A0:green"
    session_config._set_target_by_pvss(pvss)
    trace = TraceAPI(root_dir=npk_root)

    # Create the session
    session = trace.filedecode_session()

    # Start decode immediately (no need to wait for async)
    session.set_decoder_parameter("MIPI_Decoder", 'startAtAsync', 'false')
    session.pvss = pvss

    # General decode (vs. raw packet decode)
    session.usecase = "CMP General decode"

    # Start the decode
    session.decode_file(file_bin)

    # Print the csv to stdout
    session.show_output()


if __name__ == "__main__":
    # Assume an NPK_ROOT envar, else use the current directory
    main(os.getenv("NPK_ROOT", './'))
