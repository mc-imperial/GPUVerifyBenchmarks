//PASS
//--local_size=[32] --num_groups=[2]

typedef char base;
__kernel void kernel4(__global base *seq, __global int *table, int n, long c0)
{
  __requires(n == 64);
    long b0 = get_group_id(0);
    long t0 = get_local_id(0);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    #define max(x,y)    ((x) > (y) ? (x) : (y))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (10))) & (((4) * (n)) >= ((c0) + (2))));
      if (n + 7 >= c0 || (c0 >= n + 8 && 4 * ((n + 2 * c0 + 1) / 3) >= 3 * c0 + 2))
        for (long c1 = max(32 * b0, 32 * b0 + 1048576 * floord(-n - 96 * b0 + c0 - 95, 3145728) + 1048576); c1 < (c0 + 2) / 4; c1 += 1048576)
          if (t0 + c1 >= 2 && n + 3 * t0 + 3 * c1 + 1 >= c0 && c0 >= 4 * t0 + 4 * c1 + 2) {
            // shared
            table[(-4 * t0 + c0 - 4 * c1 - 2) * n + (-3 * t0 + c0 - 3 * c1 - 2)] = ((table[(-4 * t0 + c0 - 4 * c1 - 2) * n + (-3 * t0 + c0 - 3 * c1 - 2)] >= (table[(-4 * t0 + c0 - 4 * c1 - 1) * n + (-3 * t0 + c0 - 3 * c1 - 3)] + (((seq[-4 * t0 + c0 - 4 * c1 - 2] + seq[-3 * t0 + c0 - 3 * c1 - 2]) == 3) ? 1 : 0))) ? table[(-4 * t0 + c0 - 4 * c1 - 2) * n + (-3 * t0 + c0 - 3 * c1 - 2)] : (table[(-4 * t0 + c0 - 4 * c1 - 1) * n + (-3 * t0 + c0 - 3 * c1 - 3)] + (((seq[-4 * t0 + c0 - 4 * c1 - 2] + seq[-3 * t0 + c0 - 3 * c1 - 2]) == 3) ? 1 : 0)));
          }
      __function_wide_invariant(__write_implies(table, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) >= (0))) & ((((__write_offset_bytes(table)) / (sizeof(int))) % (n)) >= (((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (2)))) & ((n) >= ((((__write_offset_bytes(table)) / (sizeof(int))) % (n)) + (1)))) & ((((3) * ((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) + (c0)) == (((4) * (((__write_offset_bytes(table)) / (sizeof(int))) % (n))) + (2)))) & (((((((32) * (b0)) + (t0)) + ((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) - (((__write_offset_bytes(table)) / (sizeof(int))) % (n))) % (1048576)) == (0))));
      __function_wide_invariant(__read_implies(table, ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) >= (0))) & ((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) >= (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (2)))) & ((n) >= ((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) + (1)))) & ((((3) * ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) + (c0)) == (((4) * (((__read_offset_bytes(table)) / (sizeof(int))) % (n))) + (2)))) & (((((((32) * (b0)) + (t0)) + ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) - (((__read_offset_bytes(table)) / (sizeof(int))) % (n))) % (1048576)) == (0))) | ((((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) >= (1))) & ((n) >= ((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) + (2)))) & ((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) >= ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)))) & ((((3) * ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) + (c0)) == (((4) * (((__read_offset_bytes(table)) / (sizeof(int))) % (n))) + (9)))) & ((((((((32) * (b0)) + (t0)) + ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) - (((__read_offset_bytes(table)) / (sizeof(int))) % (n))) + (1048574)) % (1048576)) == (0)))));
      __function_wide_invariant(__read_implies(seq, (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((n) >= ((((__read_offset_bytes(seq)) / (sizeof(base))) % (n)) + (1)))) & ((c0) >= ((((__read_offset_bytes(seq)) / (sizeof(base))) % (n)) + (8)))) & ((((4) * (((__read_offset_bytes(seq)) / (sizeof(base))) % (n))) + (2)) >= (c0))) & ((((((((96) * (b0)) + ((3) * (t0))) + (((__read_offset_bytes(seq)) / (sizeof(base))) % (n))) - (c0)) + (2)) % (3145728)) == (0))) | (((((((((32) * (b0)) + (t0)) >= (0)) & ((((32) * (b0)) + (t0)) <= (1048575))) & ((((__read_offset_bytes(seq)) / (sizeof(base))) % (n)) >= (0))) & ((c0) >= ((((__read_offset_bytes(seq)) / (sizeof(base))) % (n)) + (10)))) & (((4) * (n)) >= ((((3) * (((__read_offset_bytes(seq)) / (sizeof(base))) % (n))) + (c0)) + (2)))) & ((((((((128) * (b0)) + ((4) * (t0))) + (((__read_offset_bytes(seq)) / (sizeof(base))) % (n))) - (c0)) + (2)) % (4194304)) == (0)))));
    }
}
