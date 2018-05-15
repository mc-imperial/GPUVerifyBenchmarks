//PASS
//--local_size=[32] --num_groups=[32]

__kernel void kernel0(__global int *path, int n, long c0, long c1)
{
  __requires(n == 1024);
  __requires(c0 < n); // Possible overflow in __requires
  __requires(c1 < 2 * n - 1); // Possible overflow in __requires
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define min(x,y)    ((x) < (y) ? (x) : (y))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((((n) <= (2147483647)) & ((n) >= ((c0) + (1)))) & ((c0) >= (0))) & (((2) * (n)) >= ((c1) + (2)))) & ((c1) >= (0)));
      for (long c2 = max(32 * b0, 32 * b0 + 1048576 * floord(-n - 32 * b0 + c1 - 31, 1048576) + 1048576); c2 <= min(n - 1, c1); c2 += 1048576)
        if (n >= t0 + c2 + 1 && n + t0 + c2 >= c1 + 1 && c1 >= t0 + c2) {
          // shared
          path[(-t0 + c1 - c2) * n + (t0 + c2)] = ((path[(-t0 + c1 - c2) * n + (t0 + c2)] < (path[(-t0 + c1 - c2) * n + c0] + path[c0 * n + (t0 + c2)])) ? path[(-t0 + c1 - c2) * n + (t0 + c2)] : (path[(-t0 + c1 - c2) * n + c0] + path[c0 * n + (t0 + c2)]));
        }
      __function_wide_invariant(__write_implies(path, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__write_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) + (1)))) & (((((__write_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) >= (0))) & ((n) >= ((((__write_offset_bytes(path)) / (sizeof(int))) % (n)) + (1)))) & ((((__write_offset_bytes(path)) / (sizeof(int))) % (n)) >= (0))) & ((c1) == (((((__write_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) + (((__write_offset_bytes(path)) / (sizeof(int))) % (n))))) & ((((((32) * (b0)) + (t0)) - (((__write_offset_bytes(path)) / (sizeof(int))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(path, ((((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) >= (0))) & ((n) >= ((((__read_offset_bytes(path)) / (sizeof(int))) % (n)) + (1)))) & ((((__read_offset_bytes(path)) / (sizeof(int))) % (n)) >= (0))) & ((c1) == (((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) + (((__read_offset_bytes(path)) / (sizeof(int))) % (n))))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(path)) / (sizeof(int))) % (n))) % (1048576)) == (0))) | (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(path)) / (sizeof(int))) % (n)) + (1)))) & ((((__read_offset_bytes(path)) / (sizeof(int))) % (n)) >= (0))) & ((c0) == ((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)))) & (((n) + (((__read_offset_bytes(path)) / (sizeof(int))) % (n))) >= ((c1) + (1)))) & ((c1) >= (((__read_offset_bytes(path)) / (sizeof(int))) % (n)))) & ((((((32) * (b0)) + (t0)) - (((__read_offset_bytes(path)) / (sizeof(int))) % (n))) % (1048576)) == (0)))) | (((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= (((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) + (1)))) & (((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)) >= (0))) & ((c0) == (((__read_offset_bytes(path)) / (sizeof(int))) % (n)))) & (((n) + ((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n))) >= ((c1) + (1)))) & ((c1) >= ((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n)))) & (((((((32) * (b0)) + (t0)) + ((((__read_offset_bytes(path)) / (sizeof(int))) / (n)) % (n))) - (c1)) % (1048576)) == (0)))));
    }
}
