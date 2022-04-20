#ifndef __GZIP_HEADER_DATA_HPP__
#define __GZIP_HEADER_DATA_HPP__

#include <iomanip>
#include <iostream>
#include <string>

//
// States for parsing the GZIP header
//
enum class GzipHeaderState {
  MagicNumber,
  CompressionMethod,
  Flags,
  Time,
  ExtraFlags,
  OS,
  Errata,
  Filename,
  CRC,
  Comment,
  SteadyState
};

//
// Stores the GZIP header data
//
struct GzipHeaderData {
  GzipHeaderData() {
    magic[0] = 0;
    magic[1] = 0;
    compression_method = 0;
    flags = 0;
    time[0] = 0;
    time[1] = 0;
    time[2] = 0;
    time[3] = 0;
    os = 0;
    filename[0] = '\0';
    crc[0] = 0;
    crc[1] = 0;
  }

  unsigned short MagicNumber() const {
    return ((unsigned short)(magic[0]) << 8) | (unsigned short)(magic[1]);
  }

  unsigned short CRC() const {
    return ((unsigned short)(crc[0]) << 8) | (unsigned short)(crc[1]);
  }

  unsigned int Time() const {
    unsigned int time_u = 0;
    for (int i = 0; i < 4; i++) {
      time_u |= ((unsigned int)(time[i]) << (8 * i));
    }
    return time_u;
  }

  std::string Filename() const {
    std::string ret;
    int i = 0;
    while (i < 256 && filename[i] != '\0') {
      ret.push_back(filename[i]);
      i++;
    }
    return ret;
  }

  std::string OS() const {
    switch (os) {
      case 0:
        return "FAT";
      case 1:
        return "Amiga";
      case 2:
        return "VMS";
      case 3:
        return "Unix";
      case 4:
        return "VM/CMS";
      case 5:
        return "Atari TOS";
      case 6:
        return "HPFS";
      case 7:
        return "Macintosh";
      case 8:
        return "Z-System";
      case 9:
        return "CP/M";
      case 10:
        return "TOPS-20";
      case 11:
        return "NTFS";
      case 12:
        return "Acorn RISCOS";
      case 13:
        return "FAT";
      default:
        return "Unknown";
    }
  }

  unsigned char magic[2];
  unsigned char compression_method;
  unsigned char flags;
  unsigned char time[4];
  unsigned char os;
  unsigned char filename[256];
  unsigned char crc[2];
};

std::ostream& operator<<(std::ostream& os, const GzipHeaderData& hdr_data) {
  std::ios_base::fmtflags save_flags;
  os << "GZIP Header Data\n";

  // magic number
  save_flags = os.flags();
  os << std::hex << std::setw(4) << std::setfill('0') << "Magic Number: 0x"
     << hdr_data.MagicNumber() << "\n";
  os.flags(save_flags);

  // compression method
  os << "Compression method: "
     << ((hdr_data.compression_method == 8) ? "Supported" : "Not Supported")
     << "\n";

  // flags
  os << std::hex << std::setw(4) << std::setfill('0') << "Flags: 0x"
     << (unsigned short)(hdr_data.flags) << "\n";
  os.flags(save_flags);

  // time
  os << "Time: " << hdr_data.Time() << "\n";

  // OS
  os << "OS: " << hdr_data.OS() << "\n";

  // filename
  os << "Filename: " << hdr_data.Filename() << "\n";

  // CRC
  os << std::hex << std::setw(4) << std::setfill('0') << "CRC: 0x"
     << hdr_data.CRC() << "\n";

  os.flags(save_flags);

  // ensure we restore flags
  os.flags(save_flags);
  return os;
}

#endif  // __GZIP_HEADER_DATA_HPP__