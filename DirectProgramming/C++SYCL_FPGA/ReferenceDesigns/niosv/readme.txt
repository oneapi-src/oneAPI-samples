Author:  JCJB
Date:    5/25/2023

Regtest notes:

This design requires the following ARC resources to run:

```
acdskit/23.1/115,perl/5.8.8,regutils/22.2/4,testutils/22.2/94,sycl/rel/20230629,gcc/7.4.0/2,cygwin/2.9.0,msvc/2019,aclsycltest/2023.2/20230725.5478,rocket/main/20230706.5287,hldregutil/2023.2/20230706.5285,hld_regtest_files/rel/2023.2/20230721.5455,oneapi_samples/rel/2023.2/20230704.5241,adapt/19.1/367,python/2.7.13,modelsim_se/2023.2,gdb/8.3/2,cmake/3.24.0
```

Pay special attention to modelsim_se-lic and cmake/3.24.0 (the NIOS V tools require a newer CMake than oneAPI uses)

you can run this design within ARC using the following command:

```
aocl do quartus_sh -t build_and_sim_testsystem.tcl 
```

```
arc submit modelsim_se-lic priority=90 group=opencl -- aocl do quartus_sh -t build_and_sim_testsystem.tcl 
```

Here is an ARC job: https://psg-sc-arc.sc.intel.com/arc/dashboard/reports/show_job/772341414

You can tell if the simulation succeeded by looking for this text in the [output transcript](https://psg-sc-arc.sc.intel.com/p/psg/data/whitepau/job/20230801/0400/772341414/stdout.txt):

```
# Initializing memory contents for pd_system_tb.pd_system_inst.code_data_ram.code_data_ram.altera_syncram_component.initialize_mem_contents with                                                                                                                                                                                                                                           code_data_ram_init.ver
#               990000: INFO: pd_system_tb.pd_system_inst_reset_bfm.pd_system_inst_reset_bfm.reset_deassert: Reset deasserted
# Test design for the simple DMA kernel
# 
# Test will initialize 256 incrementing four byte unsigned integers, have the accelerator DMA copy the data to a destination and then check the destination for correctness.
# Test Pass:  All the data at the destination matches the source.
# Software will now exit.
```

This design does not have any timing constraints or pin assignments so do not synthesize the design without including those first.  This design contains a JTAG UART so if it is ported
to a board ensure the JTAG I/O are contrainted properly.  The input reset path can be safely cut since the reset block inside the Platform Designer system will synchronize the reset
for you.

The design has the following directory structure:

/nios_v_test_system                - Top level
   /custom_ip                      - Directory to hold custom IP (place custom IP cores here)
   /ip                             - .ip files for all IP inside platform designer system (generated directory)
   /kernels                        - Directory containing oneAPI kernels (oneAPI kernel sources should be placed here)
   /pd_system                      - RTL comprising the Platform Designer system (generated directory)
   /pd_system_tb                   - Platform Designer test bench system (it instantiates pd_system.qsys and wraps it with BFMs, is a generated directory)
   /simulation_files               - Files to help automate co-simulating hardware and software
   /software                       - Nios V sample software with build script (application and BSP will generated from build script)
   components.ipx                  - IP search path file that extends the search path to the /custom_ip and /kernels directories
   pd_system.qsys                  - Platform Designer system (including Nios V, JTAG UART, 1MB of on-chip RAM)
   test_system.qpf                 - Quartus project file
   test_system.qsf                 - Quartus settings file


The software directory for the Nios V/g core (ensure you are using Quartus 23.1 Pro or later) has the following structure:

/software                          - Top level of the software directory
   /hello_world                    - A simple hello-world program that you can use to test your simulation environment before involving oneAPI kernels
   /simple_dma_test                - Test software that interacts with the DMA kernel in a cosimulation environment
      /app                         - Application directory where the Nios V executible (.elf) and memory initialization (.hex) files will be generated
      /bsp                         - BSP directory where all the Nios V drivers, library, and HAL files will be generated (do not touch the contents here)
      /src                         - Source directory for simple_dma_test
      software_build.sh            - Build script that creates the /app and /bsp directories and populates them.  Edit the top of this file to port it to other projects.


--------------------------
| Tool environment setup |
--------------------------

This design leverages Platform Designer, Questasim Intel edition, and oneAPI 2023.1.0.  You might not have all the environment variables set to use these various
tools so before attempting to use this design it is recommended to have the following directories in your path environment variable:

<QUARTUS_INSTALL_DIR>/qsys/bin
<QUARTUS_INSTALL_DIR>/questa_fe/bin

To use the Nios V and oneAPI compilers you should also source the following environment shells:

<ONEAPI_INSTALL_DIR>/setvars.sh
<QUARTUS_INSTALL_DIR>/niosv/bin/niosv-shell



-------------------------------------
| Running hello-world in Simulation |
-------------------------------------

It is recommend to test your design environment to ensure Nios V cosimulation is possible before attempting to add the kernel hardware.  Perform the following steps to
recreate the design from the provided sources:

1)  Generate the Platform Designer testbench:
  i)   Navigate to the design top level directory
  ii)  Open the Platform Designer system:  qsys-edit ./pd_system.qsys
  iii) Select the "Generate" toolbar menu option --> select "Generate Testbench System"
  iv)  When the Generate dialog box appears select "Generate" and wait for the testbench system to finish generating
  v)   Exit Platform Designer
  
2)  Open a Nios V shell:
  i)   Navigate to <QUARTUS_INSTALL_DIR>/niosv/bin
  ii)  Launch the Nios V shell:  ./niosv-shell
  
3)  Build the hello-world software example:
  i)   Navigate to the /software/hello_world directory
  ii)  Run software buildscript:  

    ```bash
    quartus_sh -t software_build.tcl
    ```
       This script copies the hex file from software compilation to the testbench simulation directory
      a)   The hex file lives at source location:  /software/hello_world/app/build/code_data_ram_init.hex
      b)   The testbench simulation directory: (for Questasim) lives at /pd_system_tb/pd_system_tb/sim/mentor

4)  Run the simulation:
  i)   Open vsim from <QUARTUS_INSTALL_DIR>/questa_fe/bin
  ii)  Change directory to the simulation_files directory
         a) File --> Change Directory
         b) Choose the design directory: /simulation_files
  iii) Load simulator setup file:
         a) File --> Load --> Macro File
         b) Select run_nios.tcl
  iv)   (optional) Load waveform script to automatically add Nios V and on-chip RAM to the waveform
         a) File --> Load --> Macro File
         b) Navigate to /simulation_files directory
         c) Select "wave.do"
  v)  Start the simulation:
         a) In transcript window run the command "run 2ms" to start the simulation
         b) Wait for the transcript window to print "Hello from Nios V"


------------------------------------
| Running simple-dma in Simulation |
------------------------------------

1)  Run oneAPI environment script:
  i)   Navigate to the oneAPI installation directory
  ii)  Source setvars.sh:  source setvars.sh

2)  Generate the oneAPI kernel:
  i)   Navigate to the kernel build directory at /kernels/simple_dma/build
  ii)  Invoke the kernel compilation:  icpx -fsycl -fintelfpga -I$INTELFPGAOCLSDKROOT/include ../src/simple_dma.cpp -Xshardware -fsycl-link=early -Xstarget=Arria10 -o simple_dma.o
  iii) After the oneAPI compiler is complete there should be a directory called simple_dma.prj present

3)  Add DMA kernel to Platform Designer system (it will be automatically found by components.ipx in the design top directory):
  i)   Navigate to the design top level directory
  ii)  Open the Platform Designer system:  qsys-edit ./pd_system.qsys
  iii) Add the kernel by selecting from the IP list:  oneAPI --> simple_dma_di
  iv)  Rename the kernel "simple_dma_accelerator" so that the name the Nios V software expects matches
  v)   Connect the clock and reset to the appropriate kernel ports
  vi)  Export the device_exception_bus of the kernel by double clicking the export column and giving it a name
  vii) Connect the kernel interrupt port to the Nios V interrupt receiver and assign it's IRQ number to 1 (any number besides 0 which is used by the UART)
  viii)Connect the Nios V data master to the kernel "csr_ring_root_avs" port.  Ensure the kernel's base address is assigned betweeen 0x0010_0000 and 0x001F_FFFF)
  ix)  Connect the kernel read and write masters to the "code_data_ram" on-chip memory
  x)   Keep Platform Designer open to perform subsequent steps

4)  Generate the Platform Designer testbench:
  i)   Select the "Generate" toolbar menu option --> select "Generate Testbench System"
  ii)  When the Generate dialog box appears select "Generate" and wait for the testbench system to finish generating
  iii) Exit Platform Designer

5)  Open a Nios V shell:
  i)   Navigate to <QUARTUS_INSTALL_DIR>/niosv/bin
  ii)  Launch the Nios V shell:  ./niosv-shell

6)  Build the hello-world software example:
  i)   Navigate to the /software/hello_world directory
  ii)  Run software buildscript:  

    ```bash
    quartus_sh -t software_build.tcl
    ```
       This script copies the hex file from software compilation to the testbench simulation directory
      a)   The hex file lives at source location:  /software/hello_world/app/build/code_data_ram_init.hex
      b)   The testbench simulation directory: (for Questasim) lives at /pd_system_tb/pd_system_tb/sim/mentor

7)  Run the simulation:
  i)   Open vsim from <QUARTUS_INSTALL_DIR>/questa_fe/bin
  ii)  Change directory to the simulation_files directory
         a) File --> Change Directory
         b) Choose the design directory: /simulation_files
  iii) Load simulator setup file:
         a) File --> Load --> Macro File
         b) Select run_nios.tcl
  iv)   (optional) Load waveform script to automatically add Nios V and on-chip RAM to the waveform
         a) File --> Load --> Macro File
         b) Navigate to /simulation_files directory
         c) Select "wave.do"
  v)  Start the simulation:
         a) In transcript window run the command "run 10ms" to start the simulation
         b) Wait for the transcript window to print "Software will now exit"



