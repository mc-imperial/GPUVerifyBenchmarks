//PASS
//--local_size=[1] --num_groups=[1]

typedef char base;
__kernel void kernel1(__global int *table, int n, long c0)
{
  __requires(n == 64);

    #define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    {
      __requires((((n) <= (2147483647)) & ((c0) >= (6))) & (((n) + (4)) >= (c0)));
      // shared
      table[(c0 - 6) * n + (c0 - 5)] = ((table[(c0 - 6) * n + (c0 - 5)] >= table[(c0 - 5) * n + (c0 - 6)]) ? table[(c0 - 6) * n + (c0 - 5)] : table[(c0 - 5) * n + (c0 - 6)]);
      __function_wide_invariant(__write_implies(table, ((((__write_offset_bytes(table)) / (sizeof(int))) % (n)) == (((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (1))) & ((c0) == (((((__write_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (6)))));
      __function_wide_invariant(__read_implies(table, (((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) == (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (1))) & ((c0) == (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (6)))) | ((((((__read_offset_bytes(table)) / (sizeof(int))) % (n)) + (1)) == ((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n))) & ((c0) == (((((__read_offset_bytes(table)) / (sizeof(int))) / (n)) % (n)) + (5))))));
    }
}
