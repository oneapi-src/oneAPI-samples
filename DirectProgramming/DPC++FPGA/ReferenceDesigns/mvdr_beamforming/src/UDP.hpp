#ifndef __UDP_HPP__
#define __UDP_HPP__

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include <opae/access.h>
#include <opae/enum.h>
#include <opae/mmio.h>
#include <opae/properties.h>
#include <opae/utils.h>

using namespace std::chrono;

#define OPENCL_AFU_ID "3a00972e-7aac-41de-bbd1-3901124e8cda"

// The address offsets for the various CSRs related to the
// UDP offload engine on the FPGA
#define CSR_BASE_ADR 0x30000
#define CSR_FPGA_MAC_ADR (CSR_BASE_ADR + 0x00)
#define CSR_FPGA_IP_ADR (CSR_BASE_ADR + 0x08)
#define CSR_FPGA_UDP_PORT (CSR_BASE_ADR + 0x10)
#define CSR_FPGA_NETMASK (CSR_BASE_ADR + 0x18)
#define CSR_HOST_MAC_ADR (CSR_BASE_ADR + 0x20)
#define CSR_HOST_IP_ADR (CSR_BASE_ADR + 0x28)
#define CSR_HOST_UDP_PORT (CSR_BASE_ADR + 0x30)
#define CSR_PAYLOAD_PER_PACKET (CSR_BASE_ADR + 0x38)
#define CSR_CHECKSUM_IP (CSR_BASE_ADR + 0x40)
#define CSR_RESET_REG (CSR_BASE_ADR + 0x48)
#define CSR_STATUS_REG (CSR_BASE_ADR + 0x50)

#define DEST_UDP_PORT 34543
#define CHECKSUM_IP 43369

// constants
constexpr size_t kUDPDataSize = 4096;                            // bytes
constexpr size_t kUDPHeaderSize = 2;                             // bytes
constexpr size_t kUDPTotalSize = kUDPDataSize + kUDPHeaderSize;  // bytes

// setting IP/gateway/netmask to PAC
void SetupPAC(unsigned long fpga_mac_adr, char *fpga_ip_adr,
              unsigned int fpga_udp_port, char *fpga_netmask,
              unsigned long host_mac_adr, char *host_ip_adr,
              unsigned int host_udp_port) {
  fpga_properties filter = NULL;
  fpga_token afc_token;
  fpga_handle afc_handle;
  fpga_guid guid;
  uint32_t num_matches;

  fpga_result res = FPGA_OK;
  if (uuid_parse(OPENCL_AFU_ID, guid) < 0) {
    fprintf(stderr, "Error parsing guid '%s'\n", OPENCL_AFU_ID);
    std::terminate();
  }

  // Look for AFC with MY_AFC_ID
  if ((res = fpgaGetProperties(NULL, &filter)) != FPGA_OK) {
    fprintf(stderr, "Error: creating properties object");
    std::terminate();
  }

  if ((res = fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: setting object type");
    std::terminate();
  }

  if ((res = fpgaPropertiesSetGUID(filter, guid)) != FPGA_OK) {
    fprintf(stderr, "Error: setting GUID");
  }

  if ((res = fpgaEnumerate(&filter, 1, &afc_token, 1, &num_matches)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: enumerating AFCs\n");
  }

  if (num_matches < 1) {
    fprintf(stderr, "AFC not found.\n");
    res = fpgaDestroyProperties(&filter);
    std::terminate();
  }

  // Open AFC and map MMIO
  if ((res = fpgaOpen(afc_token, &afc_handle, FPGA_OPEN_SHARED)) != FPGA_OK) {
    fprintf(stderr, "Error: opening AFC\n");
    std::terminate();
  }

  if ((res = fpgaMapMMIO(afc_handle, 0, NULL)) != FPGA_OK) {
    fprintf(stderr, "Error: mapping MMIO space\n");
    std::terminate();
  }

  // Reset AFC
  // if ((res = fpgaReset(afc_handle)) != FPGA_OK) {
  //  fprintf(stderr, "Error: resetting AFC\n");
  //  std::terminate();
  //}

  using namespace std::chrono_literals;

  // MAC reset
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_RESET_REG, 0x7)) != FPGA_OK) {
    fprintf(stderr, "Error: writing RST\n");
    std::terminate();
  }
  std::this_thread::sleep_for(1ms);

  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_RESET_REG, 0x0)) != FPGA_OK) {
    fprintf(stderr, "Error: writing RST\n");
    std::terminate();
  }
  std::this_thread::sleep_for(1ms);

  //
  // UOE register settings. These registers are not reset even after
  // fpgaClose().
  //
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_FPGA_MAC_ADR, fpga_mac_adr)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: writing FPGA MAC CSR\n");
    std::terminate();
  }
  unsigned long tmp1 = htonl(inet_addr(fpga_ip_adr));
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_FPGA_IP_ADR, tmp1)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: writing FPGA IP CSR\n");
    std::terminate();
  }
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_FPGA_UDP_PORT,
                             (unsigned long)fpga_udp_port)) != FPGA_OK) {
    fprintf(stderr, "Error: writing FPGA UDP port CSR\n");
    std::terminate();
  }
  unsigned long tmp2 = htonl(inet_addr(fpga_netmask));
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_FPGA_NETMASK, tmp2)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: writing FPGA netmask CSR\n");
    std::terminate();
  }
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_HOST_MAC_ADR, host_mac_adr)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: writing HOST MAC CSR\n");
    std::terminate();
  }
  unsigned long tmp3 = htonl(inet_addr(host_ip_adr));
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_HOST_IP_ADR, tmp3)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: writing HOST IP CSR\n");
    std::terminate();
  }
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_HOST_UDP_PORT,
                             (unsigned long)host_udp_port)) != FPGA_OK) {
    fprintf(stderr, "Error:writing HOST UDP port CSR\n");
    std::terminate();
  }
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_PAYLOAD_PER_PACKET,
                             (unsigned long)kUDPDataSize)) != FPGA_OK) {
    fprintf(stderr, "Error: writing payload per packet CSR\n");
    std::terminate();
  }
  if ((res = fpgaWriteMMIO64(afc_handle, 0, CSR_CHECKSUM_IP,
                             (unsigned long)CHECKSUM_IP)) != FPGA_OK) {
    fprintf(stderr, "Error: writing checksum IP CSR\n");
    std::terminate();
  }

  // Read status register
  unsigned long read_tmp;
  if ((res = fpgaReadMMIO64(afc_handle, 0, CSR_STATUS_REG, &read_tmp)) !=
      FPGA_OK) {
    fprintf(stderr, "Error: reading status CSR\n");
    std::terminate();
  }
  printf("Reading back status register from UDP offload engine: %012lx\n",
         read_tmp);

  // Unmap MMIO space
  if ((res = fpgaUnmapMMIO(afc_handle, 0)) != FPGA_OK) {
    fprintf(stderr, "Error: unmapping MMIO space\n");
    std::terminate();
  }
  // Release accelerator
  if ((res = fpgaClose(afc_handle)) != FPGA_OK) {
    fprintf(stderr, "Error: closing AFC\n");
    std::terminate();
  }

  // Destroy token
  if ((res = fpgaDestroyToken(&afc_token)) != FPGA_OK) {
    fprintf(stderr, "Error: destroying token\n");
    std::terminate();
  }
  // Destroy properties object
  if ((res = fpgaDestroyProperties(&filter)) != FPGA_OK) {
    fprintf(stderr, "Error: destroying properties object\n");
    std::terminate();
  }
}

// Send packets (from input_data) to the FPGA
void UDPSender(char *fpga_ip, unsigned int port, unsigned char *input_data,
               size_t packets, high_resolution_clock::time_point *t_in,
               unsigned long delay_us = 0) {
  int sock;
  struct sockaddr_in fpgaaddr;

  printf("SENDER: start\n");
  if ((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
    printf("ERROR: failed to open sender socket\n");
    std::terminate();
  }

  memset(&fpgaaddr, 0, sizeof(fpgaaddr));
  fpgaaddr.sin_family = AF_INET;
  fpgaaddr.sin_addr.s_addr = inet_addr(fpga_ip);
  fpgaaddr.sin_port = htons(port);

  printf("SENDER: starting to send packets\n");

  auto start = high_resolution_clock::now();
  for (int i = 0; i < packets; i++) {
    sendto(sock, input_data + i * kUDPTotalSize, kUDPTotalSize, 0,
           (struct sockaddr *)&fpgaaddr, sizeof(fpgaaddr));

    if (t_in) t_in[i] = high_resolution_clock::now();

    // optional delay of the producer to reduce rate of production
    if (delay_us > 0)
      std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  }
  auto end = high_resolution_clock::now();
  duration<double, std::milli> diff(end - start);

  double tp_mb_s = (kUDPTotalSize * packets * 1e-6) / (diff.count() * 1e-3);
  std::cout << "SENDER: throughput: " << tp_mb_s << " MB/s\n";

  close(sock);
  printf("SENDER: closed\n\n");
}

// Receive packets (into output_data) from the FPGA
void UDPReceiver(char *host_ip, unsigned int port, unsigned char *output_data,
                 size_t packets, high_resolution_clock::time_point *t_out) {
  int sock, res;
  struct ifreq ifr;
  struct sockaddr_in hostaddr;

  printf("RECEIVER: start\n");

  // create UDP socket
  if ((sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0) {
    printf("ERROR: fail to open receiver socket");
    std::terminate();
  }
  printf("RECEIVER: UDP socket created\n");

  // bind to ens2 interface only
  memset(&ifr, 0, sizeof(ifr));
  snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "ens2");
  if ((res = setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, (void *)&ifr,
                        sizeof(ifr))) < 0) {
    perror("Server-setsockopt() error for SO_BINDTODEVICE");
    printf("%s\n", strerror(errno));
    close(sock);
    std::terminate();
  }

  // Set port and IP
  memset(&hostaddr, 0, sizeof(hostaddr));
  hostaddr.sin_family = AF_INET;
  hostaddr.sin_addr.s_addr = inet_addr(host_ip);
  hostaddr.sin_port = htons(port);

  // Bind to the set port and IP
  if (bind(sock, (struct sockaddr *)&hostaddr, sizeof(hostaddr)) < 0) {
    printf("ERROR: fail to bind");
    std::terminate();
  }
  printf("RECEIVER: socket bound to port %d\n", DEST_UDP_PORT);

  // Receive data
  auto start = high_resolution_clock::now();
  for (int i = 0; i < packets; i++) {
    recv(sock, output_data + i * kUDPTotalSize, kUDPTotalSize, 0 /*flag*/);

    if (t_out) t_out[i] = high_resolution_clock::now();

    // DEBUG
    // std::cout << i << std::endl;
  }
  auto end = high_resolution_clock::now();
  duration<double, std::milli> diff(end - start);

  double tp_mb_s = (kUDPTotalSize * packets * 1e-6) / (diff.count() * 1e-3);
  std::cout << "RECEIVER: throughput: " << tp_mb_s << " MB/s\n";

  close(sock);
  printf("RECEIVER: closed\n\n");
}

unsigned char *AllocatePackets(size_t packets) {
  // allocate aligned memory
  auto ret = static_cast<unsigned char *>(
      aligned_alloc(1024, kUDPTotalSize * packets));

  // pin the memory
  mlock(ret, kUDPTotalSize * packets);

  return ret;
}

void FreePackets(unsigned char *ptr, size_t packets) {
  // unpin the memory
  munlock(ptr, kUDPTotalSize * packets);

  // free the memory
  free(ptr);
}

// convert an array of elements into packets including adding header
template <typename T>
void ToPackets(unsigned char *udp_bytes, T *data, size_t count) {
  assert(kUDPDataSize % sizeof(T) == 0);
  assert((count * sizeof(T)) % kUDPDataSize == 0);
  size_t count_per_packet = kUDPDataSize / sizeof(T);
  assert((count % count_per_packet) == 0);
  size_t iterations = count / count_per_packet;

  size_t packet_stride = kUDPDataSize + 2;
  for (int i = 0; i < iterations; i++) {
    udp_bytes[i * packet_stride] = 0xAB;
    udp_bytes[i * packet_stride + 1] = 0xCD;

    memcpy(&udp_bytes[i * packet_stride + 2], &data[i * count_per_packet],
           kUDPDataSize);
  }
}

// convert the bytes of packets into an array of elements
template <typename T>
void FromPackets(unsigned char *udp_bytes, T *data, size_t count) {
  assert((kUDPDataSize % sizeof(T)) == 0);
  assert((count * sizeof(T)) % kUDPDataSize == 0);
  size_t count_per_packet = kUDPDataSize / sizeof(T);
  assert((count % count_per_packet) == 0);
  size_t iterations = count / count_per_packet;

  size_t packet_stride = kUDPDataSize + 2;
  for (int i = 0; i < iterations; i++) {
    memcpy(&data[i * count_per_packet], &udp_bytes[i * packet_stride + 2],
           kUDPDataSize);
  }
}

// utility to parse a MAC address string
unsigned long ParseMACAddress(std::string mac_str) {
  std::replace(mac_str.begin(), mac_str.end(), ':', ' ');

  std::array<int, 6> mac_nums;

  std::stringstream ss(mac_str);

  int i = 0;
  int tmp;
  while (ss >> std::hex >> tmp) {
    mac_nums[i++] = tmp;
  }

  if (i != 6) {
    std::cerr << "ERROR: invalid MAC address string\n";
    return 0;
  }

  unsigned long ret = 0;
  for (size_t j = 0; j < 6; j++) {
    ret += mac_nums[j] & 0xFF;
    if (j != 5) {
      ret <<= 8;
    }
  }

  return ret;
}

// utility to pin a C++ thread to a specific CPU
int PinThreadToCPU(std::thread &t, int cpu_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu_id, &cpuset);
  return pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
}

#endif /* __UDP_HPP__ */
