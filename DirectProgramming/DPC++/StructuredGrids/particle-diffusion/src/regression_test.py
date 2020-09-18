# *******************************************************************
# *                            Description                          *
# *******************************************************************

# Test particle diffusion

# *******************************************************************
# *                              Usage                              *
# *******************************************************************

# python3 regression_test.py

# *******************************************************************
# *                             Run commands                        *
# *******************************************************************

import os
import argparse

testTypes=["particles", "powers_of_2_particles", "particles0260", "iterations",
           "grid_size", "all"]

# Parse CLAs
prser = argparse.ArgumentParser(description="Parse flags")

prser.add_argument("-c", "--cpu", help="Specify CPU flag for" +
                   " program execution.", type=str, default="0")
prser.add_argument("-o", "--output", help="Specify output flag" +
                   " for program execution.", type=str, default="0")
prser.add_argument("-r", "--rngSeed", help="Specify RNG seed for"
                   " program execution.", type=str, default="777")
prser.add_argument("-t", "--testType", help="Specify type of test" +
                   " to be conducted for program execution.",
                   type=str, choices=testTypes, default="all")

arguments = prser.parse_args()
cpuFlg = arguments.cpu
outputFlg = arguments.output
randSeed = arguments.rngSeed
testType = arguments.testType

claFlags = " -r " + randSeed + " -c " + cpuFlg + " -o " + outputFlg

ptest = 0
p2ptest = 0
p0260test = 0
itest = 0
gstest = 0

if testType == testTypes[0]: ptest = 1
elif testType == testTypes[1]: p2ptest = 1
elif testType == testTypes[2]: p0260test = 1
elif testType == testTypes[3]: itest = 1
elif testType == testTypes[4]: gstest = 1
elif testType == testTypes[5]:
  ptest = 1
  p2ptest = 1
  itest = 1
  gstest = 1

# Compile
print("Compiling ...\n")
print("dpcpp -fsycl -std=c++17 -O3 motionsim.cpp -o pd -lmkl_sycl -lmkl_intel_ilp64"+
          " -lmkl_sequential -lmkl_core\n")

os.system("dpcpp -fsycl -std=c++17 -O3 motionsim.cpp -o pd -lmkl_sycl -lmkl_intel_ilp64"+
          " -lmkl_sequential -lmkl_core")

# *******************************************************************
# *                     Scale Arguments and Execute Code            *
# *******************************************************************

if ptest == 1:
  print("********************VARYING PARTICLE COUNTS********************")
  print("./pd -i 10000 -p 1 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 1 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 5 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 5 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 10 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 10 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 20 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 20 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 40 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 40 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 100 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 100 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 150 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 150 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 200 21" + claFlags)
  os.system("./pd -i 10000 -p 200 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 256 -g 21" + claFlags)
  os.system("./pd -i 10000 -p 256 -g 21" + claFlags)
  print("-----------------------------------------------------\n\n")




  # takes a very long time to finish running (> 1 hr):
  # print("\n./pd -i 10000 -p 50 -g 21" + claFlags)
  # os.system("./pd -i 10000 -p 50 -g 21" + claFlags)
  # print("-----------------------------------------------------\n\n")

if p2ptest == 1:
  print("************VARYING PARTICLE COUNTS BY POWERS OF 2************")
  i = 2
  while i <= 65536:
    print("\n./pd -i 100 -p " + str(i) + " -g 21" + claFlags)
    os.system("./pd -i 100 -p " + str(i) + " -g 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    i = i * 2

  i = 256
  j = 0
  while j < 20:
    print("\n./pd -i 1000  -p " + str(i) + " -g 21" + claFlags)
    os.system("./pd -i 1000 -p " + str(i) + " -g 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    j = j + 1

if p0260test == 1:
  i = 1
  while i <= 260:
    print("\n./pd -i 100 -p " + str(i) + " -g 21" + claFlags)
    os.system("./pd -i 100 -p " + str(i) + " -g 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    i = i + 1

if gstest == 1:
  print("********************VARYING GRID SIZE********************")
  print("\n./pd -i 10000 -p 25 -g 1" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 1" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 5" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 5" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 10" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 10" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 15" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 15" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 20" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 20" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 30" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 30" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10000 -p 25 -g 40" + claFlags)
  os.system("./pd -i 10000 -p 25 -g 40" + claFlags)
  print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n./pd -i 10000 -p 25 -g 50" + claFlags)
  # os.system("./pd -i 10000 -p 25 -g 50" + claFlags)
  # print("-----------------------------------------------------\n\n")

if itest == 1:
  print("********************VARYING NUMBER OF ITERATIONS********************")
  print("\n./pd -i 1 -p 25 -g 50" + claFlags)
  os.system("./pd -i 1 -p 25 -g 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 10 -p 25 -g 50" + claFlags)
  os.system("./pd -i 10 -p 25 -g 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 100 -p 25 -g 50" + claFlags)
  os.system("./pd -i 100 -p 25 -g 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 1000 -p 25 -g 50" + claFlags)
  os.system("./pd -i 1000 -p 25 -g 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n./pd -i 5000 -p 25 -g 50" + claFlags)
  os.system("./pd -i 5000 -p 25 -g 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n./pd -i 10000 -p 25 -g 50" + claFlags)
  # os.system("./pd -i 10000 -p 25 -g 50" + claFlags)
  # print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n./pd -i 100000 -p 25 -g 50" + claFlags)
  # os.system("./pd -i 100000 -p 25 -g 50" + claFlags)
  # print("-----------------------------------------------------\n\n")

os.system("rm pd ")

# *******************************************************************
# *                               Done                              *
# *******************************************************************
