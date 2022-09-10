#pragma pack(push, 1)

// This is the data structure which is going to represent one pixel value in RGB
// format
typedef struct {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
} rgb;

// This block is only used when build for Structure of Arays (SOA) with Array
// Notation
typedef struct {
  unsigned char *blue;
  unsigned char *green;
  unsigned char *red;
} SOA_rgb;

#pragma pack(pop)
