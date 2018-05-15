//pass
//--global_size=514 --local_size=514

//0:34601376 1:514 2:514 3:33789056 4:514 5:514 6:514 7:514

#define VALTYPE float

static __attribute__((always_inline))
int
ToFlatIdx( int row, int col, int pitch )
{
    return row * pitch + col;
}

__kernel
void
CopyRect( __global VALTYPE* dest,
            int doffset,
            int dpitch,
            __global VALTYPE* src,
            int soffset,
            int spitch,
            int width,
            int height )
{
    __requires(doffset == 514);
    __requires(dpitch == 514);
    __requires(width == 514);

    int gid = get_group_id(0);
    int lid = get_local_id(0);
    int gsz = get_global_size(0);
    int lsz = get_local_size(0);
    int grow = gid * lsz + lid;

    if( grow < height )
    {
        for( int c = 0;
             __global_invariant(__write_implies(dest, __write_offset_bytes(dest)/sizeof(VALTYPE) - (doffset + grow * dpitch) < width)),
             c < width; c++ )
        {
            (dest + doffset)[ToFlatIdx(grow,c,dpitch)] = (src + soffset)[ToFlatIdx(grow,c,spitch)];
        }
    }
}
