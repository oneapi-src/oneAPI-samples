#ifndef __GZIP_HEADER_DATA_HPP__
#define __GZIP_HEADER_DATA_HPP__

#include <iomanip>
#include <iostream>
#include <string>

enum GzipHeaderState {
  MagicNumber, CompressionMethod, Flags, Time, ExtraFlags, OS, Errata, Filename,
  CRC, Comment, SteadyState
};

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

  unsigned short GetMagicHeader() {
    return ((unsigned short)(magic[0]) << 8) | (unsigned short)(magic[1]);
  }

  unsigned short GetCRC() {
    return ((unsigned short)(crc[0]) << 8) | (unsigned short)(crc[1]);
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
  os << "Header Data\n";
  // magic number
  save_flags = os.flags();
  os << std::hex << std::setw(4) << std::setfill('0') << "Magic Number: 0x"
     << (unsigned int)hdr_data.magic[0] << (unsigned int)hdr_data.magic[1]
     << "\n";
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
  // TODO: print better
  unsigned int time = 0;
  for (int i = 0; i < 4; i++) {
    time |= ((unsigned int)(hdr_data.time[i]) << (8 * i));
  }
  os << "Time: " << time << "\n";
  // OS
  std::string os_str;
  switch (hdr_data.os) {
    case 0:   os_str = "FAT"; break;
    case 1:   os_str = "Amiga"; break;
    case 2:   os_str = "VMS"; break;
    case 3:   os_str = "Unix"; break;
    case 4:   os_str = "VM/CMS"; break;
    case 5:   os_str = "Atari TOS"; break;
    case 6:   os_str = "HPFS"; break;
    case 7:   os_str = "Macintosh"; break;
    case 8:   os_str = "Z-System"; break;
    case 9:   os_str = "CP/M"; break;
    case 10:  os_str = "TOPS-20"; break;
    case 11:  os_str = "NTFS"; break;
    case 12:  os_str = "Acorn RISCOS"; break;
    case 13:  os_str = "FAT"; break;
    default:  os_str = "Unknown"; break;
  }
  os << "OS: " << os_str << "\n";
  // filename
  os << "Filename: ";
  char c = hdr_data.filename[0];
  int i = 1;
  while (c != '\0') {
    os << c;
    c = hdr_data.filename[i++];
  }
  os << "\n";
  // CRC
  unsigned short crc = ((unsigned short)(hdr_data.crc[0]) << 8) |
                       (unsigned short)(hdr_data.crc[1]);
  os << std::hex << std::setw(4) << std::setfill('0') << "CRC: 0x"
     << crc << "\n";
  os.flags(save_flags);
  
  // ensure we restore flags
  os.flags(save_flags);
  return os;
}

#endif // __GZIP_HEADER_DATA_HPP__