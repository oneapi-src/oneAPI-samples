oneAPI Trace Example
====================

Introduction
------------

The ISD component of oneAPI includes a python-based command line
interface for capturing/decoding system trace called TraceCLI.
Developers can use the TraceCLI python package for automating trace
capture and the ISD trace UI for development and manual trace capture.

What it is
----------

This project demonstrates decoding a trace file using the two methods
mentioned above (trace UI and TraceCLI).

Software requirements
---------------------

This sample works on both Linux and Windows.

This sample has been tested on Windows 10, and requires a working OneAPI
IOT Kit or OneAPI System Bring up installed.

Using the ISD Trace UI
----------------------

oneAPI is shipped with the Intel® System Debugger which provides an
Eclipse-based UI for capturing/decoding system trace.

Decoding an example trace .bin file offline with ISD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Launch the IDE
..............

Windows
Open the Intel System Debugger
Using the Eclipse Quick Access control (Ctrl 3), open the System Trace perspective

................................

.. figure:: ./_traceimages/image-20200417112519429.png
   :alt: image-20200417112519429

Use the System Trace wizard
...........................

.. figure:: ./_traceimages/image-20200417112603072.png
   :alt: image-20200417112603072

Enter a project name
....................

.. figure:: ./_traceimages/image-20200417115209456.png
   :alt: image-20200417115209456

Select a connection
...................

.. figure:: ./_traceimages/image-20200417115225732.png
   :alt: image-20200417115225732

Select '10th Gen Intel® Core Processor (Comet Lake) ...' and 'Intel(R) DCI USB Debug Class'
.........................................................................................

.. figure:: ./_traceimages/image-20200417115253978.png
   :alt: image-20200417115253978

Since this is a file decode, uncheck 'Update provider configuration for current connection' as well as 'Connect on finish'
..........................................................................................................................

.. figure:: ./_traceimages/image-20200417115316277.png
   :alt: image-20200417115316277

Provide a trace configuration name
..................................

.. figure:: ./_traceimages/image-20200417115329153.png
   :alt: image-20200417115329153

Import a trace capture
......................

.. figure:: ./_traceimages/image-20200417115423377.png
   :alt: image-20200417115423377

Select an example bin file (<isd install folder>/system_trace/examples/input/mipi_aet_fake_trace.bin.bin)
....................................................

.. figure:: ./_traceimages/image-20200417115643548.png
   :alt: image-20200417115643548

Select the imported capture for decoding
........................................

.. figure:: ./_traceimages/image-20200417115806272.png
   :alt: image-20200417115806272

CMP General decode

.. figure:: ./_traceimages/image-20200417115848009.png
   :alt: image-20200417115848009

A 'MessageView001' will open showing decoded trace
..................................................

.. figure:: ./_traceimages/image-20200417115934739.png
   :alt: image-20200417115934739
   :width: 150 %

Using the TraceCLI
------------------

TraceCLI has three usage models (console, file decode, and streaming)


.. code-block:: console

    > intel_tracecli  --help
    usage: intel_tracecli [-h] [-v] [--pvss PVSS] [--target TARGET]
                          [--usecase USECASE] [--transport TRANSPORT]
                          {console,decode,stream} ...

    Intel TraceCLI Version 1.2003.826.200
    Copyright Intel Corporation All rights reserved

    positional arguments:
      {console,decode,stream}
        console             Run interactive mode
        decode              Decode a trace capture file
        stream              Capture and decode traces

Running the example
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    %ISS_PYTHON3_BIN% tracecli_example.py

    > $ISS_PYTHON3_BIN/tracecli_example.py
    Intel TraceCLI Version 1.2015.469.100
    Copyright Intel Corporation All rights reserved

    Using installation at C:\Program Files (x86)\inteloneapi\system_debugger\2021.1-beta06\\system_trace



    Basic usage guideline for file decode:
      1. session = trace.filedecode_session()
      2. session.interactive_setup()
      3. session.decode_file('ipc_trace_test.bin')

    Basic usage guideline for streaming:
      1. session = trace.stream_session()
      2. session.interactive_setup()
      3. session.start_stream_capture()
      4. session.enable_trace()
      5. session.disable_trace()
      6. session.stop_stream_capture()

    Other options (Examples):
    - session.set_decoder_parameter('MIPI_Decoder', 'startAtAsync', 'false')
    - session.csv_columns.extend(['MasterID','ChannelID','payload','Summary','PacketType'])

    Info: MIPI STP Decoder Trace Statistics [instance: mipi]
    |  Master  | Channel  | Packets  | Protocol |
    |         0|         0|         0|      TSCU|
    |        24|         0|    662486| UNDEFINED|
    |        24|         1|    285638|AET_CORE_0_THREAD_0|
    |        24|         2|    284187|AET_CORE_0_THREAD_1|
    |        24|         3|    284861|AET_CORE_1_THREAD_0|
    |        24|         4|    284333|AET_CORE_1_THREAD_1|
    |        24|         5|    283629|AET_CORE_2_THREAD_0|
    |        24|         6|    284214|AET_CORE_2_THREAD_1|
    |        24|         7|    283199|AET_CORE_3_THREAD_0|
    Trace does not contain sync packet
    End of MIPI STP Decoder Trace Statistics

    "Time","Source","Summary"
    "[000]0000:00:00.000000000000","AET_CORE_0_THREAD_1","Power Entry (C0,GV) due to OTHER_THD"
    "[000]0000:00:00.000211333333","AET_CORE_1_THREAD_1","Power Exit (Ratio=0x17)"
    "[000]0000:00:00.000328583333","AET_CORE_1_THREAD_0","INT(0xEF)"
    "[000]0000:00:00.000450208333","AET_CORE_2_THREAD_0","Exception(#DE)"
    "[000]0000:00:00.000609458333","AET_CORE_0_THREAD_0","INT(0x30)"
    "[000]0000:00:00.000731958333","AET_CORE_0_THREAD_0","Exception(#DE)"
    "[000]0000:00:00.000850083333","AET_CORE_1_THREAD_1","IN(0x00000021)"
    "[000]0000:00:00.000927083333","AET_CORE_0_THREAD_1","IN(0x00000021)=0x000000EA"
    "[000]0000:00:00.000984833333","AET_CORE_3_THREAD_0","OUT(0x00000021)=0x000000EB"
    "[000]0000:00:00.001046958333","AET_CORE_0_THREAD_1","OUT(0x00000020)=0x00000060"
    "[000]0000:00:00.001783833333","AET_CORE_1_THREAD_0","IRET"
    "[000]0000:00:00.001792916667","AET_CORE_1_THREAD_0","IRET"
    "[000]0000:00:00.001800791667","AET_CORE_0_THREAD_0","IRET"
    "[000]0000:00:00.001804875000","AET_CORE_2_THREAD_1","Exception(#DE)"
    "[000]0000:00:00.001811000000","AET_CORE_2_THREAD_0","IRET"
    "[000]0000:00:00.001815250000","AET_CORE_2_THREAD_0","Exception(#DE)"
    ...
