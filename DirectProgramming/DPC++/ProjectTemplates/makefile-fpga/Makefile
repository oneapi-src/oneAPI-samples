TARGET_EMU    = fpga_emu
TARGET_HW     = fpga_hardware
TARGET_REPORT = fpga_report.a

SRCS     = src/main.cpp
OBJS     = $(SRCS:.cpp=.o)
ETS      = $(SRCS:.cpp=.d)

CXX      = dpcpp
CXXFLAGS = -std=c++17

.PHONY: build build_emu build_hw report run_emu run_hw clean run
.DEFAULT_GOAL := build_emu


# Intel-supported FPGA cards 
FPGA_DEVICE_A10 = intel_a10gx_pac:pac_a10
FPGA_DEVICE_S10 = intel_s10sx_pac:pac_s10
FPGA_DEVICE     = $(FPGA_DEVICE_A10)

# Compile flags
EMULATOR_FLAGS  = -fintelfpga -DFPGA_EMULATOR
HARDWARE_FLAGS  = -fintelfpga -Xshardware -Xsboard=$(FPGA_DEVICE)
REPORT_FLAGS    = $(HARDWARE_FLAGS) -fsycl-link


# Build for FPGA emulator
build: build_emu
build_emu: $(TARGET_EMU)

$(TARGET_EMU): $(SRCS)
	$(CXX) $(CXXFLAGS) $(EMULATOR_FLAGS) -o $@ $^ 

# Generate FPGA optimization report (without compiling all the way to hardware)
report: $(TARGET_REPORT)

$(TARGET_REPORT): $(SRCS)
	$(CXX) $(CXXFLAGS) $(REPORT_FLAGS) -o $@ $^ 

# Build for FPGA hardware
build_hw: $(TARGET_HW)

$(TARGET_HW): $(SRCS)
	$(CXX) $(CXXFLAGS) $(HARDWARE_FLAGS) -fintelfpga -o $@ $^ 

# Run on the FPGA emulator
run: run_emu
run_emu: $(TARGET_EMU)
	./$(TARGET_EMU)

# Run on the FPGA card
run_hw: $(TARGET_HW)
	./$(TARGET_HW)

# Clean all
clean:
	-$(RM) $(OBJS) $(TARGET_EMU) $(TARGET_HW) $(TARGET_REPORT) *.d
	-$(RM) -R *.prj
