CXX := dpcpp
CXXFLAGS = -O2 -g -std=c++17

SRC := src/vector-add-buffers.cpp
USM_SRC := src/vector-add-usm.cpp

.PHONY: fpga_emu run_emu fpga_emu_usm run_emu_usm clean 

fpga_emu: vector-add-buffers.fpga_emu
fpga_emu_usm: vector-add-usm.fpga_emu_usm

hw: vector-add-buffers.fpga
hw_usm: vector-add-usm.fpga

report: vector-add-buffers_report.a
report_usm: vector-add-usm_report.a_usm

vector-add-buffers.fpga_emu: $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga $^ -o $@ -DFPGA_EMULATOR=1
vector-add-usm.fpga_emu_usm: $(USM_SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga $^ -o $@ -DFPGA_EMULATOR=1


a.o: $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1
a_usm.o: $(USM_SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1	

vector-add-buffers.fpga: a.o
	$(CXX) $(CXXFLAGS) -fintelfpga $^ -o $@ -Xshardware
vector-add-usm.fpga: a_usm.o
	$(CXX) $(CXXFLAGS) -fintelfpga $^ -o $@ -Xshardware

run_emu: vector-add-buffers.fpga_emu
	./vector-add-buffers.fpga_emu
run_emu_usm: vector-add-usm.fpga_emu_usm
	./vector-add-usm.fpga_emu_usm


run_hw: vector-add-buffers.fpga
	./vector-add-buffers.fpga
run_hw_usm: vector-add-usm.fpga
	./vector-add-usm.fpga	

dev.o: $(SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1
dev_usm.o: $(USM_SRC)
	$(CXX) $(CXXFLAGS) -fintelfpga -c $^ -o $@ -DFPGA=1



vector-add-buffers_report.a: dev.o
	$(CXX) $(CXXFLAGS) -fintelfpga -fsycl-link $^ -o $@ -Xshardware
vector-add-usm_report.a_usm: dev_usm.o
	$(CXX) $(CXXFLAGS) -fintelfpga -fsycl-link $^ -o $@ -Xshardware


clean:
	rm -rf *.o *.d *.out *.mon *.emu *.aocr *.aoco *.prj *.fpga_emu *.fpga_emu_buffers vector-add-buffers.fpga  vector-add-usm.fpga *.a
