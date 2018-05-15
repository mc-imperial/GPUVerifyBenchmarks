#ifndef COMMON_CU__
#define COMMON_CU__ 1
// Children are labeled as ACGT$
const int basecount = 5;

// Note: max pixel size is 16 bytes

const unsigned char DNA_A = 'A';
const unsigned char DNA_C = 'B';
const unsigned char DNA_G = 'C';
const unsigned char DNA_T = 'D';
const unsigned char DNA_S = 'E';

// 4 bytes
struct TextureAddress
{
  union
  {
    unsigned int data;

    struct
    {
	  unsigned short x;
      unsigned short y;
    };
  };
};

// Store the start, end coordinate of node, and $link in 1 pixel
struct PixelOfNode
{
  union
  {
    ulong4 data;
    struct
    {
      int start;
      int end;
      TextureAddress childD;
      TextureAddress suffix;
    };
  };
};

// Store the ACGT links in 1 pixel
struct PixelOfChildren
{
  union
  {
    ulong4 data;
	TextureAddress children[4];
  };
};

#define FORWARD   0x0000
#define REVERSE   0x8000
#define FRMASK    0x8000
#define FRUMASK   0x7FFF

// IMPERIAL EDIT: definition from mummergpu.h
struct MatchCoord
{
  unsigned int node; // match node
  short edge_match_length;  // number of missing characters UP the parent edge
};

// IMPERIAL EDIT: common definitions from mummergpu_kernel.cu
#ifdef __DEVICE_EMULATION__no
#define VERBOSE 1
#define XPRINTF(...)  printf(__VA_ARGS__)
#else
#define XPRINTF(...)  do{}while(0)
#endif

texture<ulong4, 2, cudaReadModeElementType> nodetex;
texture<ulong4, 2, cudaReadModeElementType> childrentex;
texture<char, 2, cudaReadModeElementType> reftex;

__device__ void set_result(const TextureAddress& cur,
					   MatchCoord* result, 
					   int edge_match_length,
                       int qry_match_len,
                       int min_match_len,
                       int rc)
{
  if (qry_match_len > min_match_len)
  {
    int blocky = cur.y & 0x1F;
    int bigy = cur.y >> 5;
    int bigx = (cur.x << 5) + blocky;
    int nodeid = bigx + (bigy << 17);

    edge_match_length |= rc;
    result->node = nodeid;
    result->edge_match_length = edge_match_length;
  }
}

__device__ char getRef(int refpos)
{
  int bigx = refpos & 0x3FFFF;
  int bigy = refpos >> 18; 
  int y = (bigy << 2) + (bigx & 0x3); 
  int x = bigx >> 2; 
  return tex2D(reftex, x, y);
}

__device__ char rc(char c)
{
  switch(c)
  {
    case 'A': return 'T';
    case 'C': return 'G';
    case 'G': return 'C';
    case 'T': return 'A';
    case 'q': return '\0';
    default:  return c;
  };
}

#endif
